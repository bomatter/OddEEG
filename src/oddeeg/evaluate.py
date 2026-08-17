import math
from pathlib import Path
from pprint import pprint

import pandas as pd
import torch
import tyro
import yaml
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm

from oddeeg.datasets import create_dataset
from oddeeg.datasets.dataset_factory import get_dataset_info, get_supported_tasks
from oddeeg.metrics import build_metrics
from oddeeg.models import create_model
from oddeeg.tasks import get_metrics, get_target_mapping
from oddeeg.perturbations import apply_perturbations
from oddeeg.utils import construct_output_dir, construct_results_path


class WrappedModel(ModelWrapper):
    def __init__(self, model):
        super().__init__(None)
        self.model = model

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        if t.dim() == 0:
            t = t.repeat(x.shape[0])
        return self.model(timesteps=t, x=x, **extras)


def log_p0(x: torch.Tensor) -> torch.Tensor:
    """Compute log probability of standard Gaussian prior N(0, I)."""
    flat = x.flatten(start_dim=1)
    D = flat.shape[1]
    log_2pi = math.log(2 * math.pi)
    return -0.5 * ((flat**2).sum(dim=1) + D * log_2pi)


def compute_power_spectrum(x: torch.Tensor) -> torch.Tensor:
    """Compute the power spectrum by flattening all non-batch dims into a single signal.

    Args:
        x (torch.Tensor): Input tensor, batch on dim 0.

    Returns:
        torch.Tensor: Power spectrum of shape [batch_size, N//2 + 1] where N is the
            flattened signal length.
    """
    return torch.abs(torch.fft.rfft(x.flatten(start_dim=1), dim=-1)) ** 2


def compute_odin_score(
    model: torch.nn.Module,
    x: torch.Tensor,
    temperature: float = 1000.0,
    noise_magnitude: float = 0.0014,
) -> torch.Tensor:
    """Compute ODIN score for a batch of inputs."""
    with torch.enable_grad():
        x = x.clone().detach().requires_grad_(True)

        model.zero_grad()
        logits = model(x) / temperature
        pseudo_labels = logits.argmax(dim=1)

        loss = torch.nn.functional.cross_entropy(logits, pseudo_labels)
        loss.backward()

        x_hat = x - noise_magnitude * torch.sign(x.grad)

    with torch.no_grad():
        logits_hat = model(x_hat) / temperature
        probs_hat = torch.nn.functional.softmax(logits_hat, dim=1)
        odin_score = probs_hat.max(dim=1)[0]

    return odin_score


def compute_ash_score(
    model: torch.nn.Module,
    x: torch.Tensor,
    percentile: float = 65.0,
) -> torch.Tensor:
    """Compute ASH-S (Activation Shaping - Scale) score.
    The bottom percentile of activations are zeroed, and the remainder
    are rescaled to preserve the original activation sum. Energy score
    is then computed on the resulting logits (Djurisic et al., 2023).
    """
    base_model = model.model if hasattr(model, "model") else model

    if hasattr(base_model, "prediction_head"):
        target_layer = base_model.prediction_head
    else:
        raise ValueError("Unsupported model architecture for ASH. Ensure the model has a 'prediction_head' layer.")

    def ash_hook(module, args):
        x_in = args[0]
        x_flat = x_in.view(x_in.shape[0], -1)
        threshold = torch.quantile(x_flat, percentile / 100.0, dim=1, keepdim=True)
        threshold = threshold.view(x_in.shape[0], *[1] * (x_in.dim() - 1))

        # ASH-S: zero out the bottom percentile, then rescale to preserve total activation sum
        x_pruned = torch.where(x_in >= threshold, x_in, torch.zeros_like(x_in))
        s1 = x_in.sum(dim=list(range(1, x_in.dim())), keepdim=True)
        s2 = x_pruned.sum(dim=list(range(1, x_in.dim())), keepdim=True)
        x_shaped = x_pruned * (s1 / (s2 + 1e-8))
        return (x_shaped,)

    handle = target_layer.register_forward_pre_hook(ash_hook)

    try:
        with torch.no_grad():
            logits = model(x)
            ash_score = -torch.logsumexp(logits, dim=1)
    finally:
        handle.remove()

    return ash_score


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    task: str,
    device: str = None,  # "cuda" or "cpu"; auto-detect if None
    # Solver parameters
    step_size=0.01,
    method="euler",
    exact_divergence=False,
    # Optional prefix for metric names
    prefix: str | None = None,  # e.g. "val" or "test"
):
    """Evaluate the model on the given data loader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): The data loader for evaluation data.
        device (torch.device): The device to run the evaluation on.

    Returns:
        (pd.DataFrame, pd.DataFrame): A tuple containing two DataFrames:
            - The first DataFrame contains evaluation metrics.
            - The second DataFrame contains the predictions for each sample.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_mapping = get_target_mapping(task)
    classes = list(target_mapping.keys())

    # Ensure classes are sorted for consistent indexing
    classes = sorted(classes)
    assert classes == list(range(len(classes))), f"Classes must be consecutive integers starting from 0, got {classes}"

    model = WrappedModel(model).to(device)
    model.eval()

    solver = ODESolver(velocity_model=model)

    # Initialize metrics
    metric_tracker = build_metrics(
        metrics=get_metrics(task), device=str(device), prefixes=[prefix] if prefix is not None else []
    )

    predictions = []
    with torch.no_grad():
        for x_1, y, info in tqdm(data_loader):
            if x_1.dim() == 2:
                x_1 = x_1.unsqueeze(0)

            x_1, y = x_1.to(device), y.to(device)
            batch_size = x_1.shape[0]

            # Compute log likelihoods for all classes (in sorted order)
            log_likelihoods = []
            for c in classes:
                _, log_likelihood = solver.compute_likelihood(
                    x_1=x_1,
                    log_p0=log_p0,
                    method=method,
                    step_size=step_size,
                    exact_divergence=exact_divergence,
                    y=torch.full((batch_size,), c, dtype=torch.long, device=device),
                )
                log_likelihoods.append(log_likelihood)

            # Stack and get predictions
            log_likelihoods = torch.stack(log_likelihoods, dim=1)  # Shape: [batch_size, num_classes]
            probs = torch.nn.functional.softmax(log_likelihoods, dim=1)
            preds = log_likelihoods.argmax(dim=1)

            # Update metrics
            metric_tracker.update(probs, y)

            # Store predictions for DataFrame (iterate over batch)
            for i in range(batch_size):
                # Format log likelihoods with class names for this sample
                log_lik_dict = {f"log_likelihood_{target_mapping[c]}": log_likelihoods[i, c].item() for c in classes}

                # Extract info for this sample
                sample_info = {k: v[i].item() if isinstance(v, (torch.Tensor)) else v for k, v in info.items()}
                predictions.append(
                    {
                        **sample_info,
                        "target": target_mapping[y[i].item()],
                        "pred": target_mapping[preds[i].item()],
                        "max_log_likelihood": log_likelihoods[i].max().item(),
                        **log_lik_dict,
                    }
                )

    df_predictions = pd.DataFrame(predictions)
    df_metrics = pd.DataFrame([{k: v.item() for k, v in metric_tracker.compute().items()}])

    return df_metrics, df_predictions


def evaluate_discriminative(
    model: torch.nn.Module,
    data_loader: DataLoader,
    task: str | None = None,
    device: str = None,  # "cuda" or "cpu"; auto-detect if None
    # Optional prefix for metric names
    prefix: str | None = None,  # e.g. "val" or "test"
):
    """Evaluate a discriminative model on the given data loader.

    Args:
        model (torch.nn.Module): The discriminative model to evaluate.
        data_loader (DataLoader): The data loader for evaluation data.
        task (str | None): The task name (e.g., 'normality', 'age'). If None,
            only OOD scores are computed (no metrics, no target/pred columns).
        device (str): The device to run the evaluation on.
        prefix (str): Optional prefix for metric names.

    Returns:
        (pd.DataFrame | None, pd.DataFrame): A tuple containing:
            - Metrics DataFrame (None when task is None).
            - Predictions DataFrame with OOD scores.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if task is not None:
        target_mapping = get_target_mapping(task)
        metric_tracker = build_metrics(
            metrics=get_metrics(task), device=str(device), prefixes=[prefix] if prefix is not None else []
        )

    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating discriminative model ({prefix or 'eval'})"):
            x = batch[0]
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = x.to(device)
            batch_size = x.shape[0]

            logits = model(x)
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            normalised_entropy = torch.special.entr(probs).sum(dim=-1) / math.log(probs.size(-1))
            msp = probs.max(dim=1)[0]
            energy = -torch.logsumexp(logits, dim=1)
            odin = compute_odin_score(model, x)
            ash = compute_ash_score(model, x)

            if task is not None:
                y = batch[1].to(device)
                info = batch[2]
                metric_tracker.update(probs.detach(), y)
            else:
                info = batch[-1]

            for i in range(batch_size):
                sample_info = {k: v[i].item() if isinstance(v, torch.Tensor) else v[i] for k, v in info.items()}
                record = {
                    **sample_info,
                    "max_logit": logits[i].max().item(),
                    "normalised_entropy": normalised_entropy[i].item(),
                    "msp": msp[i].item(),
                    "energy": energy[i].item(),
                    "odin": odin[i].item(),
                    "ash": ash[i].item(),
                }
                if task is not None:
                    record["target"] = target_mapping[y[i].item()]
                    record["pred"] = target_mapping[preds[i].item()]
                predictions.append(record)

    df_predictions = pd.DataFrame(predictions)
    if task is not None:
        df_metrics = pd.DataFrame([{k: v.item() for k, v in metric_tracker.compute().items()}])
    else:
        df_metrics = None

    return df_metrics, df_predictions


def evaluate_likelihoods(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str = None,  # "cuda" or "cpu"; auto-detect if None
    # Solver parameters
    step_size=0.01,
    method="euler",
    exact_divergence=False,
    # Optional prefix for metric names
    prefix: str | None = None,  # e.g. "val" or "test"
):
    """Evaluate unconditional log-likelihoods for samples.

    Args:
        model (torch.nn.Module): The unconditional flow matching model.
        data_loader (DataLoader): The data loader for evaluation data.
        device (str): The device to run the evaluation on.
        step_size: Step size for ODE solver.
        method: ODE solver method.
        exact_divergence: Whether to compute exact divergence.
        prefix: Optional prefix for output naming.

    Returns:
        pd.DataFrame: DataFrame containing log-likelihoods and sample metadata.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WrappedModel(model).to(device)
    model.eval()

    solver = ODESolver(velocity_model=model)

    predictions = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Computing likelihoods ({prefix or 'eval'})"):
            x_1 = batch[0].to(device)
            info = batch[-1]  # Use last item for compatibility with datasets that return label as second item

            if x_1.dim() == 2:
                x_1 = x_1.unsqueeze(0)

            batch_size = x_1.shape[0]

            # Compute unconditional log-likelihood (no class conditioning)
            x_source, log_likelihood = solver.compute_likelihood(
                x_1=x_1,
                log_p0=log_p0,
                method=method,
                step_size=step_size,
                exact_divergence=exact_divergence,
                y=None,  # No class conditioning
            )

            source_log_likelihood = log_p0(x_source)
            log_determinant = log_likelihood - source_log_likelihood

            power_spectrum = compute_power_spectrum(x_source)
            ps_cv = power_spectrum.std(dim=-1) / (power_spectrum.mean(dim=-1) + 1e-8)

            # Store predictions for DataFrame (iterate over batch)
            for i in range(batch_size):
                x_source_i = x_source[i].flatten().cpu().numpy()
                ad_stat = stats.anderson(x_source_i, dist="norm").statistic

                sample_info = {k: v[i].item() if isinstance(v, torch.Tensor) else v[i] for k, v in info.items()}
                predictions.append(
                    {
                        **sample_info,
                        "log_likelihood": log_likelihood[i].item(),
                        "source_log_likelihood": source_log_likelihood[i].item(),
                        "log_determinant": log_determinant[i].item(),
                        "anderson_darling_statistic": ad_stat,
                        "ps_cv": ps_cv[i].item(),
                    }
                )

    df_predictions = pd.DataFrame(predictions)

    return df_predictions


def evaluate_from_config(
    config: Path,  # path to a config.yml file
    split_pick: str = "test",  # which split to evaluate on, e.g. "val" or "test"
    device: str | None = None,
    batch_size: int = 128,
    num_workers: int = 4,
    # Solver parameters
    step_size: float | None = 0.01,
    method: str = "euler",
    exact_divergence: bool = False,
    # Perturbation parameters
    perturbation_sfreq: float | None = None,
    perturbation_channel_shuffle: float | None = None,
    perturbation_lowpass_hz: float | None = None,
    perturbation_highpass_hz: float | None = None,
    perturbation_reref_scheme: str | None = None,
    # Cross-dataset evaluation
    eval_dataset_name: str | None = None,  # evaluate on a different dataset than the training one
    # Output
    save_results: bool = True,  # save results to disk
):
    """Evaluate a model based on the provided configuration."""

    # Capture evaluation parameters for later inclusion in metrics
    eval_params = {
        k: v for k, v in locals().items() if k not in ("config", "device", "batch_size", "num_workers", "save_results")
    }
    config_path = Path(config)

    if not config_path.is_file():
        raise ValueError(f"Config at path {config_path} is not a valid file.")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("Running evaluation with the following configurations:")
    pprint(eval_params)

    # Resolve cross-dataset evaluation
    eval_task = config.get("task")
    if eval_dataset_name is not None:
        if eval_task not in get_supported_tasks(eval_dataset_name):
            eval_task = None
        data_split = "default"
    else:
        eval_dataset_name = config.get("dataset_name", "TUAB")
        data_split = config.get("data_split", "default")

    dataset = create_dataset(
        dataset_name=eval_dataset_name,
        task=eval_task,
        preprocessing=config.get("preprocessing", "minimal"),
        sfreq=config.get("sfreq", 100),
        segment_length=config.get("segment_length", 2.0),
        data_split=data_split,
        return_meta_data=True,
        pick=split_pick,
        scratch_root=False,
    )

    # Apply perturbations (modifies in place)
    apply_perturbations(
        datasets=dataset,
        perturbation_sfreq=perturbation_sfreq,
        perturbation_channel_shuffle=perturbation_channel_shuffle,
        perturbation_lowpass_hz=perturbation_lowpass_hz,
        perturbation_highpass_hz=perturbation_highpass_hz,
        perturbation_reref_scheme=perturbation_reref_scheme,
        channels=get_dataset_info(eval_dataset_name)["channels"],
        sfreq=config.get("sfreq", 100),
    )

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    model = create_model(
        model_name=config.get("model_name", "TCN"),
        dataset_name=config.get("dataset_name", "TUAB"),
        task=config.get("task", "normality"),
        training_mode=config.get("training_mode", "discriminative"),
        n_times=int(config.get("segment_length", 2.0) * config.get("sfreq", 100)),
        checkpoint=construct_output_dir(config).joinpath("model_checkpoint.pth"),
        device=device,
        channel_mult=config.get("channel_mult", [1, 2, 4, 8]),
        attention_resolutions=config.get("attention_resolutions", [16, 8]),
    )

    training_mode = config.get("training_mode", "discriminative")

    if training_mode == "flow_matching":
        if eval_task is not None:
            df_metrics, df_predictions = evaluate(
                model=model,
                data_loader=data_loader,
                task=eval_task,
                device=device,
                step_size=step_size,
                method=method,
                exact_divergence=exact_divergence,
                prefix=split_pick,
            )
        else:
            df_predictions = evaluate_likelihoods(
                model=model,
                data_loader=data_loader,
                device=device,
                step_size=step_size,
                method=method,
                exact_divergence=exact_divergence,
                prefix=split_pick,
            )
            df_metrics = None
    elif training_mode == "discriminative":
        df_metrics, df_predictions = evaluate_discriminative(
            model=model,
            data_loader=data_loader,
            task=eval_task,
            device=device,
            prefix=split_pick,
        )
    else:
        raise ValueError(f"Unknown training mode: {training_mode}")

    # Add config and evaluation parameters to metrics
    if df_metrics is not None:
        df_metrics = df_metrics.assign(
            **{
                k: ",".join(str(i) for i in v) if isinstance(v, (list, tuple)) else v
                for k, v in {**config, **eval_params}.items()
            }
        )

    if save_results:
        if df_metrics is not None:
            metrics_path = construct_results_path(config=config_path, result_type="metrics", **eval_params)
            df_metrics.to_csv(metrics_path, index=False)
            print(f"Saved metrics to {metrics_path}")
        predictions_path = construct_results_path(config=config_path, result_type="predictions", **eval_params)
        df_predictions.to_csv(predictions_path, index=False)
        print(f"Saved predictions to {predictions_path}")

    return df_metrics, df_predictions


def main():
    tyro.cli(evaluate_from_config)


if __name__ == "__main__":
    main()
