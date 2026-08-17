import warnings

import torch
import torchaudio.functional as AF
from scipy.signal import butter, filtfilt
from torch.utils.data import Subset
from torchvision.transforms import Compose


class SFreqPerturbation:
    """Resample signal to different sampling rate.

    This perturbation resamples the signal and then crops/pads to maintain
    the original number of samples. This means:
    - Higher sampling rates: cropping end of signal
    - Lower sampling rates: zero-padding at end of signal

    Args:
        sfreq: Target sampling frequency in Hz
        orig_sfreq: Original sampling frequency in Hz (default: 100)
    """

    def __init__(self, sfreq: float, orig_sfreq: float = 100):
        self.orig_sfreq = orig_sfreq
        self.sfreq = sfreq

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply resampling to input tensor.

        Args:
            x: Input tensor of shape [channels, time]

        Returns:
            Resampled tensor with same shape as input
        """
        if self.orig_sfreq == self.sfreq:
            return x

        orig_length = x.shape[-1]

        # Resample
        resampled = AF.resample(x, orig_freq=int(self.orig_sfreq), new_freq=int(self.sfreq))

        # Crop or pad to original length
        new_length = resampled.shape[-1]
        if new_length > orig_length:
            resampled = resampled[..., :orig_length]
        elif new_length < orig_length:
            resampled = torch.nn.functional.pad(resampled, (0, orig_length - new_length))

        return resampled


class ChannelShufflePerturbation:
    """Shuffle a random subset of channels.

    Selects k = round(fraction * n_channels) channels at random and applies a
    random derangement (permutation with no fixed points) to their positions,
    guaranteeing that every selected channel is actually moved.

    Special cases:
    - fraction=0.0: k=0, silent no-op.
    - fraction>0 but k=1: derangement is impossible; warns and returns input unchanged.
    - fraction>0 and k>=2: random derangement is applied.

    Args:
        fraction: Fraction of channels to shuffle. Must be in [0, 1].
        seed: Optional integer seed for reproducibility.
    """

    def __init__(self, fraction: float, seed: int | None = 42):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        self.fraction = fraction
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel shuffle to input tensor.

        Args:
            x: Input tensor of shape [channels, time]

        Returns:
            Tensor with same shape as input, with a subset of channels permuted
            among themselves (guaranteed derangement: no channel stays in place).
        """
        n_channels = x.shape[0]
        k = round(self.fraction * n_channels)

        if k == 0:
            return x

        if k == 1:
            warnings.warn(
                f"ChannelShufflePerturbation: fraction={self.fraction} yields k=1 channel "
                f"for n_channels={n_channels}. A derangement requires k>=2; "
                "returning input unchanged.",
                UserWarning,
                stacklevel=2,
            )
            return x

        # Select k channels to shuffle
        selected = torch.randperm(n_channels, generator=self.rng)[:k]

        # Draw a random derangement of k elements via rejection sampling
        identity = torch.arange(k)
        while True:
            perm = torch.randperm(k, generator=self.rng)
            if not (perm == identity).any():
                break

        out = x.clone()
        out[selected] = x[selected[perm]]
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(fraction={self.fraction})"


class LowpassPerturbation:
    """Apply a zero-phase Butterworth low-pass filter to the signal.

    Attenuates frequency content above the cutoff, simulating limited hardware
    bandwidth, aggressive anti-aliasing, or a low-pass preprocessing step.
    Lower cutoff = more severe perturbation (more high-frequency content removed).

    Args:
        cutoff_hz: Low-pass cutoff frequency in Hz. Must be in (0, sfreq / 2).
        sfreq: Sampling frequency of the signal in Hz (default: 100).
        order: Butterworth filter order (default: 4).
    """

    def __init__(self, cutoff_hz: float, sfreq: float = 100, order: int = 4):
        if not 0 < cutoff_hz < sfreq / 2:
            raise ValueError(f"cutoff_hz must be in (0, sfreq/2) = (0, {sfreq / 2}), got {cutoff_hz}")
        self.cutoff_hz = cutoff_hz
        self.sfreq = sfreq
        self.order = order
        self._b, self._a = butter(order, cutoff_hz / (sfreq / 2), btype="low")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply zero-phase low-pass filter to input tensor.

        Args:
            x: Input tensor of shape [channels, time]

        Returns:
            Filtered tensor with same shape as input.
        """
        out = filtfilt(self._b, self._a, x.numpy(), axis=-1)
        return torch.from_numpy(out.copy()).to(x.dtype)

    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff_hz={self.cutoff_hz}, sfreq={self.sfreq}, order={self.order})"


class HighpassPerturbation:
    """Apply a zero-phase Butterworth high-pass filter to the signal.

    Attenuates frequency content below the cutoff, simulating aggressive
    high-pass filtering or removal of slow drifts and low-frequency oscillations.
    Higher cutoff = more severe perturbation (more low-frequency content removed).

    Args:
        cutoff_hz: High-pass cutoff frequency in Hz. Must be in (0, sfreq / 2).
        sfreq: Sampling frequency of the signal in Hz (default: 100).
        order: Butterworth filter order (default: 4).
    """

    def __init__(self, cutoff_hz: float, sfreq: float = 100, order: int = 4):
        if not 0 < cutoff_hz < sfreq / 2:
            raise ValueError(f"cutoff_hz must be in (0, sfreq/2) = (0, {sfreq / 2}), got {cutoff_hz}")
        self.cutoff_hz = cutoff_hz
        self.sfreq = sfreq
        self.order = order
        self._b, self._a = butter(order, cutoff_hz / (sfreq / 2), btype="high")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply zero-phase high-pass filter to input tensor.

        Args:
            x: Input tensor of shape [channels, time]

        Returns:
            Filtered tensor with same shape as input.
        """
        out = filtfilt(self._b, self._a, x.numpy(), axis=-1)
        return torch.from_numpy(out.copy()).to(x.dtype)

    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff_hz={self.cutoff_hz}, sfreq={self.sfreq}, order={self.order})"


class RereferencePerturbation:
    """Re-reference EEG data to a different scheme.

    Simulates a mismatch between the reference scheme used during training
    (assumed to be average reference) and one applied at test time. Three
    common clinical schemes are supported, with a qualitative ordering of
    increasing dissimilarity from average reference:

    - ``"linked_temporal"``: Linked temporal reference: subtract the average
      of T7 and T8 from all channels. Serves as a proxy for linked mastoids
      when M1/M2 are not in the electrode cap.
    - ``"cz"``: Monopolar vertex reference: subtract channel Cz from all
      channels. Cz itself becomes identically zero.
    - ``"bipolar"``: Longitudinal double-banana bipolar montage: 18 standard
      anteroposterior pairs plus a T7-T8 transverse pair (19 channels total).
      Each output channel is the difference between an electrode pair and the
      channel ordering follows the standard clinical sequence.

    Args:
        scheme: Referencing scheme to apply. One of ``"cz"``,
            ``"linked_temporal"``, or ``"bipolar"``.
        channels: Ordered list of channel names matching the tensor's channel
            dimension. Obtain via
            ``get_dataset_info(dataset_name)["channels"]``.
    """

    SCHEMES = {"cz", "linked_temporal", "bipolar"}

    # Standard double-banana longitudinal pairs + T7-T8 transverse (19 total).
    # Each tuple is (anode, cathode); output channel = x[anode] - x[cathode].
    BIPOLAR_PAIRS: list[tuple[str, str]] = [
        # Left lateral chain
        ("Fp1", "F7"),
        ("F7", "T7"),
        ("T7", "P7"),
        ("P7", "O1"),
        # Right lateral chain
        ("Fp2", "F8"),
        ("F8", "T8"),
        ("T8", "P8"),
        ("P8", "O2"),
        # Left parasagittal chain
        ("Fp1", "F3"),
        ("F3", "C3"),
        ("C3", "P3"),
        ("P3", "O1"),
        # Right parasagittal chain
        ("Fp2", "F4"),
        ("F4", "C4"),
        ("C4", "P4"),
        ("P4", "O2"),
        # Midline chain
        ("Fz", "Cz"),
        ("Cz", "Pz"),
        # Transverse
        ("T7", "T8"),
    ]

    def __init__(self, scheme: str, channels: list[str]):
        if scheme not in self.SCHEMES:
            raise ValueError(f"scheme must be one of {sorted(self.SCHEMES)}, got '{scheme}'")
        self.scheme = scheme
        self.channels = channels
        self._ch_idx = {ch: i for i, ch in enumerate(channels)}

        if scheme == "cz":
            if "Cz" not in self._ch_idx:
                raise ValueError("'Cz' not found in channels — cannot apply Cz reference")
            self._cz_idx = self._ch_idx["Cz"]

        elif scheme == "linked_temporal":
            for ch in ("T7", "T8"):
                if ch not in self._ch_idx:
                    raise ValueError(f"'{ch}' not found in channels — cannot apply linked temporal reference")
            self._t7_idx = self._ch_idx["T7"]
            self._t8_idx = self._ch_idx["T8"]

        elif scheme == "bipolar":
            missing = {ch for pair in self.BIPOLAR_PAIRS for ch in pair if ch not in self._ch_idx}
            if missing:
                raise ValueError(f"Channels missing for bipolar montage: {sorted(missing)}")
            self._bipolar_idx = [(self._ch_idx[a], self._ch_idx[b]) for a, b in self.BIPOLAR_PAIRS]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply re-referencing to input tensor.

        Args:
            x: Input tensor of shape [channels, time]

        Returns:
            Re-referenced tensor with same shape as input.
        """
        if self.scheme == "cz":
            return x - x[self._cz_idx : self._cz_idx + 1, :]

        elif self.scheme == "linked_temporal":
            ref = (x[self._t7_idx : self._t7_idx + 1, :] + x[self._t8_idx : self._t8_idx + 1, :]) / 2.0
            return x - ref

        else:  # bipolar
            return torch.stack([x[a] - x[b] for a, b in self._bipolar_idx], dim=0)

    def __repr__(self):
        return f"{self.__class__.__name__}(scheme='{self.scheme}')"


def apply_perturbations(
    datasets,
    perturbation_sfreq=None,
    perturbation_channel_shuffle: float | None = None,
    perturbation_lowpass_hz: float | None = None,
    perturbation_highpass_hz: float | None = None,
    perturbation_reref_scheme: str | None = None,
    channels: list[str] | None = None,
    sfreq: float = 100,
) -> None:
    """Apply perturbations to dataset(s) (modifies in place).

    Parameters
    ----------
    datasets : Dataset or list of Dataset
        Dataset(s) to apply perturbations to (may be Subset objects)
    perturbation_sfreq : float, optional
        Frequency for sfreq perturbation
    perturbation_channel_shuffle : float, optional
        Fraction of channels to shuffle (0.0 = no-op, 1.0 = full shuffle)
    perturbation_lowpass_hz : float, optional
        Low-pass cutoff frequency in Hz
    perturbation_highpass_hz : float, optional
        High-pass cutoff frequency in Hz
    perturbation_reref_scheme : str, optional
        Re-referencing scheme to apply. One of ``"cz"``, ``"linked_temporal"``,
        or ``"bipolar"``.
    channels : list[str], optional
        Ordered list of channel names matching the tensor's channel dimension.
        Required when ``perturbation_reref_scheme`` is set. Can be obtained with
        ``get_dataset_info(dataset_name)["channels"]``.
    sfreq : float
        Sampling frequency of the signal in Hz; required by filter perturbations
    """

    # Handle single dataset
    if not isinstance(datasets, list):
        datasets = [datasets]

    # Early return if no perturbations
    if not (
        perturbation_sfreq is not None
        or perturbation_channel_shuffle is not None
        or perturbation_lowpass_hz is not None
        or perturbation_highpass_hz is not None
        or perturbation_reref_scheme is not None
    ):
        return

    # Build transform
    transforms_list = []
    if perturbation_sfreq is not None:
        transforms_list.append(SFreqPerturbation(sfreq=perturbation_sfreq, orig_sfreq=sfreq))
    if perturbation_channel_shuffle is not None:
        transforms_list.append(ChannelShufflePerturbation(fraction=perturbation_channel_shuffle))
    if perturbation_lowpass_hz is not None:
        transforms_list.append(LowpassPerturbation(cutoff_hz=perturbation_lowpass_hz, sfreq=sfreq))
    if perturbation_highpass_hz is not None:
        transforms_list.append(HighpassPerturbation(cutoff_hz=perturbation_highpass_hz, sfreq=sfreq))
    if perturbation_reref_scheme is not None:
        if channels is None:
            raise ValueError("channels must be provided when perturbation_reref_scheme is set")
        transforms_list.append(RereferencePerturbation(scheme=perturbation_reref_scheme, channels=channels))

    transform = Compose(transforms_list)

    # Apply to all datasets
    for dataset in datasets:
        if isinstance(dataset, Subset):
            dataset.dataset.transform = transform
        else:
            dataset.transform = transform
