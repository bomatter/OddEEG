import torchmetrics
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.wrappers import MetricTracker


def build_metrics(
    metrics: dict,
    names: list[str] = None,
    prefixes: list[str] = [],
    device: str = "cpu",
    maximize: bool = None,
):
    """Adds metrics to the tracker."""
    new_metrics = []
    for metric_name, args in metrics.items():
        if hasattr(torchmetrics.regression, metric_name):
            metric = getattr(torchmetrics.regression, metric_name)(**args)
        elif hasattr(torchmetrics.classification, metric_name):
            metric = getattr(torchmetrics.classification, metric_name)(**args)
        elif hasattr(torchmetrics.aggregation, metric_name):
            metric = getattr(torchmetrics.aggregation, metric_name)(**args)
        else:
            raise ValueError(
                f"Metric {metric_name} not available in torchmetrics.regression, "
                + "torchmetrics.classification, or torchmetrics.aggregation."
            )
        metric.persistent(True)
        new_metrics.append(metric)

    if names is not None:
        new_metrics = {name: metric for name, metric in zip(names, new_metrics)}

    if prefixes:
        new_metrics = [MetricTracker(MetricCollection(new_metrics, prefix=p), maximize=maximize) for p in prefixes]
        for metric in new_metrics:
            metric.to(device)
            metric.increment()
        if len(new_metrics) == 1:
            new_metrics = new_metrics[0]
    else:
        new_metrics = MetricTracker(MetricCollection(new_metrics), maximize=maximize)
        new_metrics.to(device)
        new_metrics.increment()

    return new_metrics


class Tracker:
    """Wrapper class to track train and validation metrics together."""

    def __init__(
        self,
        n_batches_per_epoch: int,
        metrics: dict = None,
        early_stopping_metric=("Loss", "min"),
        device: str = "cpu",
        wandb=None,
    ):

        self.epoch = 0
        self.batch = 0
        self.n_batches_per_epoch = n_batches_per_epoch

        # Build metrics if provided, otherwise only track loss
        if metrics is not None:
            self.metrics_train, self.metrics_val = build_metrics(metrics, prefixes=["train", "val"], device=device)
        else:
            self.metrics_train = None
            self.metrics_val = None

        self.loss_train, self.loss_val = build_metrics(
            {"MeanMetric": {}}, names=["Loss"], prefixes=["train", "val"], device=device, maximize=False
        )

        # Initialise placeholders for the current and best metrics
        self.last_report = None
        self.best_metrics = None

        # Track best validation results for early stopping
        metric_names = list(metrics.keys()) if metrics is not None else []
        assert early_stopping_metric[0] in metric_names + ["Loss"], (
            f"Early stopping metric {early_stopping_metric[0]} not found in metrics: {metric_names + ['Loss']}."
        )
        assert early_stopping_metric[1] in ["min", "max"], (
            f"Early stopping metric direction {early_stopping_metric[1]} has to be 'min' or 'max'."
        )
        self.evaluations_since_improvement = 0
        self.early_stopping_metric = ("val" + early_stopping_metric[0], early_stopping_metric[1])  # Add prefix to name
        self.best_val_metric = float("inf") if self.early_stopping_metric[1] == "min" else -float("inf")

        # Note: torchmetrics currently uses an unreliable way of determining whether to apply a sigmoid.
        # It is determined on a per-batch basis and checks if the predictions are floats outside of [0, 1].
        # For consistent behaviour, we always apply sigmoid for binary classification.
        self.apply_sigmoid = False
        if metrics is not None:
            self.apply_sigmoid = any(
                [
                    hasattr(torchmetrics.classification, m)
                    and isinstance(
                        getattr(torchmetrics.classification, m)(**args),
                        torchmetrics.classification.stat_scores.BinaryStatScores,
                    )
                    for m, args in metrics.items()
                ]
            )

        self.wandb = wandb
        self.wandb_run_id = getattr(wandb, "id", None) if wandb is not None else None

    def report_train_step(self, loss, pred=None, target=None, epoch=None, batch=None):
        assert batch is None or batch == self.batch, (
            f"Inconsistent batch number. Expected {self.batch}, got {batch}. Make sure you call update every batch."
        )
        assert epoch is None or epoch == self.epoch, (
            f"Inconsistent epoch number. Expected {self.epoch}, got {epoch}. Make sure you call update every batch."
        )

        self.loss_train.update(loss)
        if self.metrics_train is not None:
            assert pred is not None and target is not None, "pred and target must be provided when metrics are tracked."
            if self.apply_sigmoid:
                pred = nn.functional.sigmoid(pred)
            self.metrics_train.update(pred, target)

        self.batch = (self.batch + 1) % self.n_batches_per_epoch
        self.epoch += 1 if self.batch == 0 else 0

    def report_val_step(self, loss, pred=None, target=None):
        self.loss_val.update(loss)
        if self.metrics_val is not None:
            assert pred is not None and target is not None, "pred and target must be provided when metrics are tracked."
            if self.apply_sigmoid:
                pred = nn.functional.sigmoid(pred)
            self.metrics_val.update(pred, target)

    def report_val_done(self):
        """Call this method after all validation batches for an evaluation have been processed."""
        self.last_report = self._compute_metrics()
        self._notify_wandb()

        current_val_metric = self.last_report[self.early_stopping_metric[0]]
        if (self.early_stopping_metric[1] == "min" and current_val_metric < self.best_val_metric) or (
            self.early_stopping_metric[1] == "max" and current_val_metric > self.best_val_metric
        ):
            self.best_val_metric = current_val_metric
            self.best_metrics = self.last_report
            self.evaluations_since_improvement = 0
        else:
            self.evaluations_since_improvement += 1

        if self.metrics_train is not None:
            self.metrics_train.increment()
        if self.metrics_val is not None:
            self.metrics_val.increment()
        self.loss_train.increment()
        self.loss_val.increment()

    def _notify_wandb(self):
        if self.wandb is not None:
            self.wandb.log(self.last_report)

    def _compute_metrics(self):
        # Collect metrics and losses into a dictionary
        metrics_dict = {
            "epoch": self.epoch,
            "batch": self.batch,
            "batch_global": self.epoch * self.n_batches_per_epoch + self.batch,
            **(self.metrics_train.compute() if self.metrics_train is not None else {}),
            **(self.metrics_val.compute() if self.metrics_val is not None else {}),
            **self.loss_train.compute(),
            **self.loss_val.compute(),
        }

        metrics_dict = {
            k: v.tolist() if k not in ["epoch", "batch", "batch_global"] else v for k, v in metrics_dict.items()
        }
        return metrics_dict

    def get_metrics(self):
        """
        Returns the last reported metrics.
        """
        return self.last_report

    def get_best_metrics(self):
        """
        Returns the metrics at the lowest validation loss.
        """
        return self.best_metrics

    def state_dict(self):
        """Return serialisable state for checkpointing."""
        return {
            "epoch": self.epoch,
            "batch": self.batch,
            "evaluations_since_improvement": self.evaluations_since_improvement,
            "best_val_metric": self.best_val_metric,
            "best_metrics": self.best_metrics,
            "last_report": self.last_report,
            "wandb_run_id": self.wandb_run_id,
        }

    def load_state_dict(self, state):
        """Restore state from a checkpoint."""
        self.epoch = state["epoch"]
        self.batch = state["batch"]
        self.evaluations_since_improvement = state["evaluations_since_improvement"]
        self.best_val_metric = state["best_val_metric"]
        self.best_metrics = state["best_metrics"]
        self.last_report = state["last_report"]
        self.wandb_run_id = state.get("wandb_run_id", None)
