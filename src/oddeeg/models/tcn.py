import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils.parametrizations import weight_norm

from oddeeg.datasets import get_dataset_info
from oddeeg.tasks import get_task_info


def create_model(
    dataset_name,
    task,
    n_times=None,
    training_mode: str = "discriminative",
    **kwargs,  # ignored
):
    if training_mode != "discriminative":
        raise ValueError(
            f"TCN model is designed for discriminative tasks and does not support training_mode='{training_mode}'."
        )

    dataset_info = get_dataset_info(dataset_name)
    n_channels = dataset_info["n_channels"]

    task_info = get_task_info(task)
    n_classes = task_info["n_outputs"]

    return TCN(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_blocks=4,
        n_filters=64,
        kernel_size=5,
        drop_prob=0.0,
        n_times=n_times,
    )


class TCN(nn.Module):
    """
    Modern implementation of Temporal Convolutional Network (TCN) (Bai et al. 2018)
    with several extensions.

    Parameters
    ----------
    n_chans: int
        Number of input channels (EEG channels)
    n_outputs: int
        Number of output classes/features
    n_blocks: int, default=3
        Number of temporal blocks in the network
    n_filters: int, default=30
        Number of output filters of each convolution
    kernel_size: int, default=5
        Kernel size of the convolutions
    drop_prob: float, default=0.0
        Dropout probability
    activation: nn.Module, default=nn.ReLU
        Activation function class to apply
    temporal_pooling: int | None | str, default=1
        Temporal pooling strategy:
        - None: No pooling, per-timestep predictions
        - int: Adaptive average pooling to specified output length
            (e.g., 1 for global pooling, 5 for 5 timesteps)
        - "fc": Fully-connected layer to collapse temporal dimension.
            n_times must be specified if temporal_pooling is "fc".
    n_times: int | None = None,
        Number of time steps in the input sequence (required if temporal_pooling is "fc").
    target_receptive_field: int | None, default=None
        If specified, adjusts dilations to cyclic/wavenet-like pattern with
        automatically determined max_dilation to achieve the desired receptive
        field (as close to, but not exceeding, target_receptive_field).
    rezero: bool, default=False
        If True, uses ReZero connections in temporal blocks.
    """

    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_blocks: int = 3,
        n_filters: int = 30,
        kernel_size: int = 5,
        drop_prob: float = 0.0,
        activation: nn.Module = nn.ReLU,
        temporal_pooling: int | None | str = 1,
        n_times: int | None = None,
        target_receptive_field: int | None = None,
        rezero: bool = False,
    ):
        super().__init__()

        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_blocks = n_blocks
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.drop_prob = drop_prob
        self.activation = activation
        self.temporal_pooling_type = temporal_pooling

        # Build temporal blocks
        self.temporal_blocks = nn.ModuleList()
        self.receptive_field = 1
        self.dilations = generate_dilation_pattern(n_blocks, kernel_size, target_receptive_field)
        for i, dilation in enumerate(self.dilations):
            n_inputs = n_chans if i == 0 else n_filters
            padding = (kernel_size - 1) * dilation

            block = TemporalBlock(
                n_inputs=n_inputs,
                n_outputs=n_filters,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                drop_prob=drop_prob,
                activation=activation,
                rezero=rezero,
            )
            self.temporal_blocks.append(block)
            self.receptive_field += 2 * (kernel_size - 1) * dilation

        # Build temporal pooling layer based on strategy
        if isinstance(self.temporal_pooling_type, int):
            self.temporal_pooling = nn.AdaptiveAvgPool1d(self.temporal_pooling_type)
        elif self.temporal_pooling_type in {"fc", None}:
            self.temporal_pooling = nn.Identity()
        else:
            raise ValueError(f"Unsupported temporal_pooling type: {self.temporal_pooling_type}")

        # Prediction head to aggregate across channels (1x1 conv for compatibility with all temporal pooling options)
        self.prediction_head = nn.Conv1d(n_filters, n_outputs, kernel_size=1)

        # For fc, temporal pooling is applied after aggregation over channels
        if self.temporal_pooling_type == "fc":
            if n_times is None:
                raise ValueError("n_times must be provided when temporal_pooling='fc'")
            output_n_times = n_times - self.receptive_field + 1
            self.fc_temporal = nn.Linear(output_n_times, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if issubclass(self.activation, nn.ReLU):
                    init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif issubclass(self.activation, nn.LeakyReLU):
                    init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                else:
                    # Fallback to Xavier for other activations
                    init.xavier_normal_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, n_channels, n_times)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_outputs, output_times)
            where output_times depends on temporal pooling strategy.
            Singleton dimensions are removed, i.e. if n_outputs = 1,
            the output shape will be (batch_size, output_times) or if
            temporal_pooling is 1, it will be (batch_size, n_outputs).
        """
        batch_size, n_channels, time_size = x.shape

        # Validate input
        if n_channels != self.n_chans:
            raise ValueError(f"Expected {self.n_chans} channels, got {n_channels}")
        if time_size < self.receptive_field:
            raise ValueError(f"Input length {time_size} is less than minimum required length {self.receptive_field}")

        # Pass through temporal blocks
        for block in self.temporal_blocks:
            x = block(x)

        # Remove time steps influenced by padding
        x = x[:, :, -(time_size - self.receptive_field + 1) :]

        # Apply temporal pooling and prediction
        x = self.temporal_pooling(x)
        x = self.prediction_head(x)

        if self.temporal_pooling_type == "fc":
            x = self.fc_temporal(x)
        elif self.temporal_pooling_type == "max_with_pos":
            x = self.max_with_pos(x)

        x = x.squeeze(-1)

        return x


class TemporalBlock(nn.Module):
    """
    Temporal block for TCN.

    Implements a residual block with dilated convolutions.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        padding: int,
        drop_prob: float,
        activation: nn.Module = nn.ReLU,
        rezero: bool = False,
    ):
        super().__init__()

        # First convolution
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=1, padding=padding, dilation=dilation)

        # Apply weight normalization
        self.conv1 = weight_norm(self.conv1)

        self.chomp1 = ChompPadding(padding)
        self.activation1 = activation()
        self.dropout1 = nn.Dropout1d(drop_prob)

        # Second convolution
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=1, padding=padding, dilation=dilation)

        # Apply weight normalization
        self.conv2 = weight_norm(self.conv2)

        self.chomp2 = ChompPadding(padding)
        self.activation2 = activation()
        self.dropout2 = nn.Dropout1d(drop_prob)

        # Skip connection (1x1 conv if input/output dims differ)
        self.skip_connection = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()
        if isinstance(self.skip_connection, nn.Conv1d):
            init.normal_(self.skip_connection.weight, 0, 0.01)

        # ReZero parameter
        self.rezero = rezero
        if self.rezero:
            self.residual_weight = nn.Parameter(torch.zeros(1))

        self.final_activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through temporal block."""
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.activation1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.activation2(out)
        out = self.dropout2(out)

        if self.rezero:
            out = out * self.residual_weight

        out = out + self.skip_connection(x)
        out = self.final_activation(out)

        return out


class ChompPadding(nn.Module):
    """Remove padding from the right side of 1D convolution output."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Remove chomp_size elements from the right."""
        if self.chomp_size > 0:
            return x[:, :, : -self.chomp_size].contiguous()
        return x


def compute_receptive_field(kernel_sizes, dilations):
    """
    Compute the receptive field for given kernel sizes and dilations in
    the TCN architecture (i.e. assuming 2 convolutions per block).

    Args:
        kernel_sizes (list of int): List of kernel sizes for each layer.
        dilations (list of int): List of dilation factors for each layer.
    """
    receptive_field = 1
    for kernel_size, dilation in zip(kernel_sizes, dilations):
        receptive_field += 2 * (kernel_size - 1) * dilation
    return receptive_field


def generate_dilation_pattern(n_blocks, kernel_size, target_receptive_field=None):
    if target_receptive_field is None:
        # Standard exponential dilation
        return [2**i for i in range(n_blocks)]

    # Wavenet/cyclic pattern with automatic max_dilation selection
    kernel_sizes = [kernel_size] * n_blocks
    candidates = [2**i for i in range(1, n_blocks + 1)]
    for candidate_max_dilation in reversed(candidates):
        current = 1
        candidate_dilation = [1]
        for _ in range(1, n_blocks):
            current *= 2
            if current > candidate_max_dilation:
                current = 1
            candidate_dilation.append(current)

        candidate_receptive_field = compute_receptive_field(kernel_sizes, candidate_dilation)

        if candidate_receptive_field <= target_receptive_field:
            return candidate_dilation
