import random
import time
from copy import deepcopy
from math import ceil
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import tyro
import wandb
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from oddeeg.datasets import create_dataset
from oddeeg.datasets.dataset import subsample_dataset
from oddeeg.evaluate import evaluate, evaluate_discriminative, evaluate_likelihoods
from oddeeg.metrics import Tracker
from oddeeg.models import create_model
from oddeeg.tasks import get_loss_function, get_metrics
from oddeeg.utils import construct_output_dir, construct_results_path, set_seed


def save_results(output_dir, config, metrics_dict, val_predictions, test_predictions, eval_params, model=None):
    """Save training results including config, metrics, and optionally model checkpoint."""

    # Save config
    with open(output_dir / "config.yml", "w") as f:
        yaml.dump(config, f)

    # Save metrics
    metrics_df = pd.DataFrame([metrics_dict]).assign(
        **{k: ",".join(str(d) for d in v) if isinstance(v, (list, tuple)) else v for k, v in config.items()}
    )
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    # Save validation and test predictions
    val_predictions.to_csv(
        construct_results_path(
            config=config, output_dir=output_dir, split_pick="val", result_type="predictions", **eval_params
        ),
        index=False,
    )
    test_predictions.to_csv(
        construct_results_path(
            config=config, output_dir=output_dir, split_pick="test", result_type="predictions", **eval_params
        ),
        index=False,
    )

    # Save model checkpoint if provided
    if model is not None:
        torch.save(
            getattr(model, "_orig_mod", model).state_dict(),
            output_dir / "model_checkpoint.pth",
        )


def save_training_checkpoint(
    output_dir,
    state,
    model,
    optimizer,
    tracker,
):
    """Save an intermediate training checkpoint for resumption.

    Args:
        output_dir: Directory to save the checkpoint to.
        state: Mutable dict of training progress (global_batch, epoch, etc.).
        model: The training model.
        optimizer: The optimizer.
        tracker: The metric tracker.
    """
    checkpoint = {
        "state": state,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "tracker_state": tracker.state_dict(),
        "rng_states": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    # Write to a temp file then rename for atomicity
    tmp_path = output_dir / "training_checkpoint.pth.tmp"
    torch.save(checkpoint, tmp_path)
    tmp_path.rename(output_dir / "training_checkpoint.pth")
    print(f"Saved training checkpoint at batch {state['global_batch']}.")


def load_training_checkpoint(
    output_dir,
    state,
    model,
    optimizer,
    tracker,
    device,
):
    """Load a training checkpoint, restoring all state in-place.

    Modifies state, model, optimizer, tracker in-place.

    Args:
        output_dir: Directory containing the checkpoint.
        state: Mutable dict to update with saved training progress.
        model: The training model.
        optimizer: The optimizer.
        tracker: The metric tracker.
        device: Device to map tensors to.

    Returns:
        True if a checkpoint was loaded, False otherwise.
    """
    ckpt_path = output_dir / "training_checkpoint.pth"
    if not ckpt_path.exists():
        return False

    print(f"Resuming from checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Restore training progress
    state.update(checkpoint["state"])

    # Restore model, optimizer, tracker
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    tracker.load_state_dict(checkpoint["tracker_state"])

    # Restore RNG states
    rng = checkpoint["rng_states"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.random.set_rng_state(rng["torch"].cpu())
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([b.cpu() for b in rng["cuda"]])

    print(
        f"Resumed from batch {state['global_batch']} "
        f"(epoch {state['epoch']}, batch-in-epoch {state['batch_in_epoch']})."
    )

    return True


def evaluate_on_val(model, dataloader, loss_function, task, tracker, training_mode, device):
    """Evaluate model on validation set."""
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating on validation set", leave=True):
            x_1 = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True) if task is not None else None
            if training_mode == "discriminative":
                out = model(x_1)
                loss = loss_function(out, y)
                tracker.report_val_step(loss=loss.item(), pred=out.detach(), target=y)
            else:  # flow_matching
                x_0 = torch.randn_like(x_1, device=device)
                t = torch.rand(len(x_1), device=device)
                x_t = (1 - t.view(-1, 1, 1)) * x_0 + t.view(-1, 1, 1) * x_1
                dx_t = x_1 - x_0
                out = model(timesteps=t, x=x_t, y=y)
                loss = loss_function(out, dx_t)
                tracker.report_val_step(loss=loss.item())

    tracker.report_val_done()
    model.train()


def train(
    # Dataset parameters
    dataset_name: str = "TUAB",
    data_split: str = "default",
    preprocessing: str = "minimal",
    segment_length: float = 2.0,  # in seconds
    # Task
    task: str | None = "normality",  # set to None for unconditional flow matching
    # Model parameters
    model_name: str = "TCN",
    channel_mult: list[int] = [1, 2, 2, 2],
    attention_resolutions: list[int] = [8],  # downsampling factors
    # Subsampling
    n_participants: int | None = None,  # subsample training data to this many participants
    n_segments: int | None = None,  # max segments per participant in subsampled training data
    segment_select: str = "random",  # segment selection method: "random" or "contiguous"
    unique_recording_per_participant: bool = True,  # pick one recording per participant when subsampling
    stratify_participants: str | None = None,  # column to stratify participant sampling by (e.g. "normality")
    # Training parameters
    training_mode: str = "discriminative",  # training mode: "flow_matching" or "discriminative"
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    batch_size: int = 128,
    max_batches: int = 50000,
    evaluation_interval: int = 500,  # in batches
    early_stopping_patience: int | None = None,  # in evaluations; deactivate if None
    early_stopping_metric: str = "Loss",  # metric to use for early stopping
    early_stopping_mode: str = "min",  # whether to maximize or minimize the early stopping metric
    restore_best: bool = True,  # restore best model after training
    # Output parameters
    output_root: str | None = None,  # will use user_config["output_root"] if None
    save_checkpoint: bool = True,  # save model checkpoint after training
    use_wandb: bool = False,
    overwrite: bool = False,  # delete existing intermediate checkpoint and start fresh
    # Other parameters
    cache: bool = False,  # whether to use caching in dataset (requires sufficient memory)
    scratch_root: str | None = None,  # uses user_config["scratch_root"] if None; set to "" to disable explicitly
    num_workers: int = 4,  # number of workers for DataLoader
    seed: int = 42,
    deterministic: bool = False,  # deterministic cuda operations (may slow down training)
    debug: bool = False,  # cap training and evaluation to a few batches for quick checks
):

    # Save configuration
    config = {
        k: v
        for k, v in locals().items()
        if k
        not in {
            "output_root",
            "save_checkpoint",
            "use_wandb",
            "overwrite",
            "cache",
            "scratch_root",
            "num_workers",
            "debug",
        }
    }

    # Validate config
    assert training_mode in {"flow_matching", "discriminative"}
    assert training_mode == "flow_matching" or task is not None, "Discriminative mode requires a task"

    print("Running training with the following configurations:")
    pprint(config)

    # Construct and create output directory
    output_dir = construct_output_dir(config=config, output_root=output_root, debug=debug)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if training has already completed (final checkpoint exists)
    if (output_dir / "model_checkpoint.pth").exists() and not overwrite:
        print("Training already completed (found model_checkpoint.pth). Skipping. Use --overwrite to retrain.")
        return None

    # Handle overwrite: remove existing intermediate checkpoint
    if overwrite and (output_dir / "training_checkpoint.pth").exists():
        print("Overwrite requested — removing existing training checkpoint.")
        (output_dir / "training_checkpoint.pth").unlink()

    # Record start time
    start_time = time.time()

    # Configure random seed and deterministic behavior
    if seed is not None:
        set_seed(seed=seed, deterministic=deterministic)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Constants
    sfreq = 100
    n_times = int(segment_length * sfreq)

    # Create dataset
    # Note: transforms are set later such that we can apply sample rejection on untransformed data
    dataset_train, dataset_val, dataset_test = create_dataset(
        dataset_name=dataset_name,
        task=task,
        preprocessing=preprocessing,
        sfreq=sfreq,
        segment_length=segment_length,
        data_split=data_split,
        pick=["train", "val", "test"],
        return_meta_data=["val", "test"],
        cache=cache,
        scratch_root=scratch_root,
    )

    # Subsample training data if specified
    if n_participants is not None or n_segments is not None:
        dataset_train = subsample_dataset(
            dataset_train,
            n_participants=n_participants,
            n_segments=n_segments,
            segment_select=segment_select,
            unique_recording_per_participant=unique_recording_per_participant,
            stratify_by=stratify_participants,
            seed=seed,
        )

    if debug:
        max_batches = 10
        evaluation_interval = 10
        n = 2 * batch_size
        dataset_train = torch.utils.data.Subset(dataset_train, range(min(n, len(dataset_train))))
        dataset_val = torch.utils.data.Subset(dataset_val, range(min(n, len(dataset_val))))
        dataset_test = torch.utils.data.Subset(dataset_test, range(min(n, len(dataset_test))))

    # Create data loaders
    dataloader_train = DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    dataloader_val = DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    dataloader_test = DataLoader(
        dataset_test,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create model
    model = create_model(
        model_name=model_name,
        dataset_name=dataset_name,
        n_times=n_times,
        task=task,
        training_mode=training_mode,
        device=device,
        compile=False,
        channel_mult=channel_mult,
        attention_resolutions=attention_resolutions,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if training_mode == "discriminative":
        loss_function = get_loss_function(task=task)
    else:  # flow_matching
        loss_function = torch.nn.MSELoss()

    # Initialize metric tracker (wandb run is set after checkpoint restore)
    tracker = Tracker(
        n_batches_per_epoch=len(dataloader_train),
        metrics=get_metrics(task) if task is not None else None,
        early_stopping_metric=(early_stopping_metric, early_stopping_mode),
        device=device,
        wandb=None,
    )

    # Mutable training state — modified in-place by load_training_checkpoint
    state = {
        "global_batch": 0,
        "epoch": 0,
        "batch_in_epoch": 0,
        "evaluations": 0,
        "training_time_offset": 0.0,
        "best_model_state": None,
    }

    # Attempt to resume from intermediate checkpoint
    resumed = load_training_checkpoint(
        output_dir=output_dir,
        state=state,
        model=model,
        optimizer=optimizer,
        tracker=tracker,
        device=device,
    )

    # Create wandb run (if enabled) — done after checkpoint restore so we can
    # retrieve the run ID from the tracker for proper wandb resumption.
    wandb_run = None

    # Create wandb run (if enabled)
    if use_wandb:
        wandb_kwargs = {}
        if resumed and tracker.wandb_run_id is not None:
            wandb_kwargs = {"id": tracker.wandb_run_id, "resume": "must"}
        wandb_run = wandb.init(
            config=config,
            project="oddeeg",
            group="-".join([dataset_name, task or "unconditional", model_name]),
            dir=output_dir,
            mode="offline",
            **wandb_kwargs,
        )
        wandb_run.watch(model)

    # Train
    global_batch = state["global_batch"]
    evaluations = state["evaluations"]
    best_model_state = state["best_model_state"]
    resume_epoch = state["epoch"]
    resume_batch_in_epoch = state["batch_in_epoch"]
    training_time_offset = state["training_time_offset"]
    stop_training = False
    model.train()
    max_epochs = ceil(max_batches / len(dataloader_train))
    for epoch in tqdm(range(resume_epoch, max_epochs), position=0, desc="Epochs ", leave=True):
        if stop_training:
            break

        for batch_idx, batch in enumerate(tqdm(dataloader_train, position=1, desc="Batches", leave=True)):
            # Skip batches already processed in a resumed epoch
            if epoch == resume_epoch and batch_idx < resume_batch_in_epoch:
                continue

            if global_batch >= max_batches:
                break

            x_1 = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True) if task is not None else None

            if training_mode == "discriminative":
                out = model(x_1)
                loss = loss_function(out, y)
            else:  # flow_matching
                x_0 = torch.randn_like(x_1, device=device)
                t = torch.rand(len(x_1), device=device)
                x_t = (1 - t.view(-1, 1, 1)) * x_0 + t.view(-1, 1, 1) * x_1
                dx_t = x_1 - x_0
                out = model(timesteps=t, x=x_t, y=y)
                loss = loss_function(out, dx_t)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_batch += 1
            tracker.report_train_step(
                loss=loss.item(),
                epoch=epoch,
                batch=batch_idx,
                pred=out.detach() if training_mode == "discriminative" else None,
                target=y if training_mode == "discriminative" else None,
            )

            # Evaluate model
            if global_batch % evaluation_interval == 0:
                evaluate_on_val(
                    model,
                    dataloader_val,
                    loss_function,
                    task,
                    tracker,
                    training_mode,
                    device,
                )
                current_metrics = tracker.get_metrics()
                pprint(current_metrics)
                evaluations += 1

                # Backup best model state
                if restore_best and tracker.evaluations_since_improvement == 0:
                    best_model_state = deepcopy(model.state_dict())

                # Early stopping
                if early_stopping_patience is not None:
                    if tracker.evaluations_since_improvement >= early_stopping_patience:
                        print(f"Early stopping triggered after {global_batch} batches.")
                        stop_training = True
                        break

                # Save intermediate training checkpoint
                state.update(
                    {
                        "global_batch": global_batch,
                        "epoch": epoch,
                        "batch_in_epoch": batch_idx + 1,
                        "evaluations": evaluations,
                        "training_time_offset": training_time_offset + (time.time() - start_time),
                        "best_model_state": best_model_state,
                    }
                )
                save_training_checkpoint(
                    output_dir=output_dir,
                    state=state,
                    model=model,
                    optimizer=optimizer,
                    tracker=tracker,
                )

    training_time = training_time_offset + (time.time() - start_time)

    if restore_best and best_model_state is not None:
        print("Restoring best model...")
        model.load_state_dict(best_model_state)

    if use_wandb:
        wandb_run.unwatch(model)
        wandb_run.finish()

    # Evaluate on validation and test sets
    if training_mode == "flow_matching":
        eval_params = {
            "method": "euler",
            "step_size": 0.01,
        }
        if task is not None:
            print("Evaluating classification performance on validation set...")
            val_metrics, val_predictions = evaluate(
                model=model,
                data_loader=dataloader_val,
                task=task,
                device=str(device),
                prefix="val",
                **eval_params,
            )
            print("Evaluating classification performance on test set...")
            test_metrics, test_predictions = evaluate(
                model=model,
                data_loader=dataloader_test,
                task=task,
                device=str(device),
                prefix="test",
                **eval_params,
            )
        else:
            print("Computing likelihoods on validation set...")
            val_predictions = evaluate_likelihoods(
                model=model,
                data_loader=dataloader_val,
                device=str(device),
                prefix="val",
                **eval_params,
            )
            val_metrics = None
            print("Computing likelihoods on test set...")
            test_predictions = evaluate_likelihoods(
                model=model,
                data_loader=dataloader_test,
                device=str(device),
                prefix="test",
                **eval_params,
            )
            test_metrics = None
    elif training_mode == "discriminative":
        eval_params = {}
        print("Evaluating discriminative model on validation set...")
        val_metrics, val_predictions = evaluate_discriminative(
            model=model,
            data_loader=dataloader_val,
            task=task,
            device=str(device),
            prefix="val",
        )
        print("Evaluating discriminative model on test set...")
        test_metrics, test_predictions = evaluate_discriminative(
            model=model,
            data_loader=dataloader_test,
            task=task,
            device=str(device),
            prefix="test",
        )

    total_time = training_time_offset + (time.time() - start_time)

    print("Final evaluation metrics:")
    metrics_dict = {
        **(tracker.get_best_metrics() if restore_best else tracker.get_metrics()),
        **(test_metrics.iloc[0].to_dict() if test_metrics is not None else {}),
        "training_time": training_time,
        "total_time": total_time,
    }
    # Note: adding val metrics through update because they are already tracked during training
    # for discriminative models and should be overwritten here with final results in case training
    # was not terminated through early stopping and could thus have changed after last evaluation.
    metrics_dict.update(val_metrics.iloc[0].to_dict() if val_metrics is not None else {})
    pprint(metrics_dict)

    # Save checkpoint and config
    save_results(
        output_dir=output_dir,
        config=config,
        metrics_dict=metrics_dict,
        val_predictions=val_predictions,
        test_predictions=test_predictions,
        eval_params=eval_params,
        model=model if save_checkpoint else None,
    )

    # Clean up intermediate checkpoint
    ckpt_path = output_dir / "training_checkpoint.pth"
    if ckpt_path.exists():
        ckpt_path.unlink()
        print("Removed intermediate training checkpoint.")

    return metrics_dict


def main():
    tyro.cli(train)


if __name__ == "__main__":
    main()
