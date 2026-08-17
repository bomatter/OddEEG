import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from joblib import Parallel, delayed
from mne_bids import BIDSPath, find_matching_paths, read_raw_bids
from torch.utils.data import Dataset, Subset
from tqdm import tqdm


PREPROCESSING = {
    "minimal": {
        "filter": {"l_freq": 1, "h_freq": 50},
    }
}


class BaseNpyDataset(Dataset):
    """
    Base class for datasets that load .npy files. Provides functionality for caching
    loaded files in memory or using a scratch directory for faster access.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        cache: bool = False,
        scratch_root: str = None,
    ):
        """
        Args:
            cache (bool, optional):
                If True, cache loaded files in memory for faster access. Defaults to False.

            scratch_root (str, optional):
                Path to a scratch directory. If provided, data will be saved to it when first loaded. If data
                is accessed in subsequent epochs, it will be loaded from the scratch directory instead of the
                original data directory. This is similar to caching, just that data is saved to a scratch directory
                instead of being kept in memory. Should be used when not enough memory is available for caching
                and data access is faster from the scratch directory than the original data directory.
        """
        self.dataset_root = Path(dataset_root)
        self.cache = cache
        self._file_cache = {} if cache else None

        # Setup scratch directory
        if scratch_root:
            if self.cache:
                warn("Caching is enabled, so scratch mode will be disabled.")
                self._scratch_dir = None
            else:
                self.setup_scratch(scratch_root)
        else:
            self._scratch_dir = None

    def setup_scratch(self, scratch_root: str | Path):
        # Set and create scratch directory
        self._scratch_dir = Path(scratch_root)
        self._scratch_dir.mkdir(parents=True, exist_ok=True)

        # Create a ref file to register scratch usage
        job_id = os.environ.get("SLURM_JOB_ID", "")  # use slurm job ID if available
        job_id += f"_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self._scratch_ref_file = self._scratch_dir / "refcounts" / f"{job_id}.ref"
        self._scratch_ref_file.parent.mkdir(parents=True, exist_ok=True)
        self._scratch_ref_file.touch(exist_ok=False)  # create ref file

        # Initialise scratch map
        self._scratch_map = {}
        print(f"Using scratch directory: {self._scratch_dir}")

    def _load_file(self, file_path: Path):
        """
        Load a file with caching and scratch directory support.

        Args:
            file_path (Path): Path to the file to load.

        Returns:
            np.ndarray: Loaded data array.
        """
        if self.cache:
            # Cache memory-mapped files for OS-level caching with multiple workers
            if file_path not in self._file_cache:
                self._file_cache[file_path] = np.load(file_path, mmap_mode="r")
            return self._file_cache[file_path]
        elif self._scratch_dir:
            # Copy file to scratch directory if not already done
            if file_path not in self._scratch_map:
                scratch_path = self._scratch_dir / file_path.relative_to(self.dataset_root)
                scratch_path.parent.mkdir(parents=True, exist_ok=True)
                with FileLock(scratch_path.with_suffix(scratch_path.suffix + ".lock")):
                    if not scratch_path.exists():
                        shutil.copy(file_path, scratch_path)
                self._scratch_map[file_path] = scratch_path
            # Use memory-mapping to avoid loading entire file into memory
            return np.load(self._scratch_map[file_path], mmap_mode="r")
        else:
            # Direct loading with memory-mapping
            return np.load(file_path, mmap_mode="r")

    def __del__(self):
        """Remove scratch directory if it was created."""
        if self._scratch_dir:
            if not self._scratch_dir.exists():  # Directory already removed
                return

            # Remove ref file to indicate that scratch directory is no longer in use
            self._scratch_ref_file.unlink(missing_ok=True)

            try:
                # Check if scratch directory is still being used by other processes
                now = datetime.now().timestamp()
                keep = False
                for ref_file in (self._scratch_dir / "refcounts").glob("*.ref"):
                    # Ignore ref files older than 31 days (2678400 seconds)
                    if now - ref_file.stat().st_mtime < 2678400:
                        keep = True
                        break
                # If no other processes are using the scratch directory, remove it
                if not keep:
                    shutil.rmtree(self._scratch_dir)
            except Exception as e:
                print(f"Error removing scratch directory: {e}")


class NpyDataset(BaseNpyDataset):
    def __init__(
        self,
        dataset_root: str | Path,
        dataset_index: str | Path | pd.DataFrame = None,
        segment_length: float = 2.0,
        target: str = None,
        target_mapping: dict = None,
        return_meta_data: bool = False,
        transform=None,
        cache: bool = False,
        scratch_root: str = None,
    ):
        """
        Args:
            dataset_root (str | Path):
                Root directory of the dataset.

            dataset_index (str | Path | pd.DataFrame, optional):
                Path to a CSV file or a DataFrame containing the following columns:
                    - 'participant_id': Unique identifier for each participant
                    - 'sfreq': Sampling frequency of the EEG recordings
                    - 'n_timepoints': Total number of timepoints in the recording
                    - 'file': Path to the data file (NumPy .npy format) relative to dataset_root
                    - If `target` is specified, a column with the same name as `target`.
                If None, will attempt to load dataset_index.csv from dataset_root.

            segment_length (float, optional):
                Length of EEG samples in seconds. Defaults to 2.0.

            target (str, optional):
                Name of the column in dataset_index to use as target labels.
                If specified, __getitem__ will return (data, target) tuples.

            target_mapping (dict, optional):
                Mapping from target values to numerical values. E.g. {"M": 0.0, "F": 1.0}.

            return_meta_data (bool, optional):
                If True, __getitem__ will additionally return information about the sample,
                such as participant_id and segment index within the recording.
                Returns a tuple of the form (data, metadata) or (data, target, metadata) if target is specified.

            transform (callable, optional):
                A callable (or composed transforms) to apply to data samples.
                Should accept and return a torch.Tensor. Use torchvision.transforms.Compose
                to combine multiple transforms.

            cache (bool, optional):
                If True, cache loaded files in memory for faster access. Defaults to False.

            scratch_root (str, optional):
                Path to a scratch directory. If provided, data will be saved to it when first loaded. If data
                is accessed in subsequent epochs, it will be loaded from the scratch directory instead of the
                original data directory. This is similar to caching, just that data is saved to a scratch directory
                instead of being kept in memory. Should be used when not enough memory is available for caching
                and data access is faster from the scratch directory than the original data directory.
        """

        super().__init__(dataset_root=dataset_root, cache=cache, scratch_root=scratch_root)

        self.segment_length = segment_length
        self.target = target
        self.target_mapping = target_mapping
        self.return_meta_data = return_meta_data
        self.transform = transform

        # Initialize dataset index after setting other properties
        # since the dataset_index setter will validate and process it
        # based on other properties
        if dataset_index is None:
            self.dataset_index = pd.read_csv(self.dataset_root / "dataset_index.csv")
        elif isinstance(dataset_index, str) or isinstance(dataset_index, Path):
            self.dataset_index = pd.read_csv(dataset_index)
        elif isinstance(dataset_index, pd.DataFrame):
            self.dataset_index = dataset_index
        else:
            raise TypeError("If provided (not None), dataset_index must be a file path (str) or a pandas DataFrame.")

    @property
    def dataset_index(self):
        return self._dataset_index

    @dataset_index.setter
    def dataset_index(self, new_index):
        # Check that required columns are available
        required_cols = ["sfreq", "n_timepoints"] + [self.target] if self.target is not None else []
        missing_cols = [col for col in required_cols if col not in new_index.columns]
        if missing_cols:
            raise ValueError(f"Dataset index missing required columns: {missing_cols}")

        # Check that sfreq is consistent across all recordings
        if not np.all(new_index["sfreq"] == new_index["sfreq"].iloc[0]):
            raise ValueError("All recordings must have the same sampling frequency (sfreq).")

        # Drop rows with NaN values in target column if target is specified
        if self.target is not None:
            initial_length = len(new_index)
            new_index = new_index.dropna(subset=[self.target]).copy()
            dropped_count = initial_length - len(new_index)
            if dropped_count > 0:
                print(f"Warning: Dropped {dropped_count} recordings with NaN values in target column '{self.target}'")

            # Check if dataset is empty after filtering
            if len(new_index) == 0:
                raise ValueError("Dataset is empty after filtering. No valid recordings found.")

        # Count number of segments for each recording for the given segment length
        new_index["n_segments"] = np.floor(
            (new_index["n_timepoints"] / new_index["sfreq"]) / self.segment_length
        ).astype(int)

        # Update dataset index after passing validation and curation
        self._dataset_index = new_index

        # Update dependent properties
        self.cumulative_segments = np.cumsum(self.dataset_index["n_segments"].values)
        self.total_segments = self.cumulative_segments[-1]

    def __len__(self):
        """Return total number of segments across all recordings."""
        return self.total_segments

    def __getitem__(self, idx):
        """
        Returns:
            Returns an EEG data segment (and if specified during dataset construction, target and metadata).
            - data (torch.Tensor): EEG data segment of shape (n_channels, n_samples).
            - target (float | str | None): Target label if specified, else None.
            - metadata (dict): Metadata including participant_id and segment index.
        """

        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self)} segments")

        # Determine recording index
        recording_idx = np.searchsorted(self.cumulative_segments, idx, side="right")

        # Determine segment index within the recording
        if recording_idx == 0:
            segment_idx = idx
        else:
            segment_idx = idx - self.cumulative_segments[recording_idx - 1]

        # Get recording metadata
        metadata = self.dataset_index.iloc[recording_idx]

        # Load data, with optional caching
        file_path = self.dataset_root / metadata["file"]
        data = self._load_file(file_path)

        # Extract segment
        t_start = int(segment_idx * self.segment_length * metadata["sfreq"])
        t_end = int(t_start + self.segment_length * metadata["sfreq"])
        data = torch.tensor(data[:, t_start:t_end])

        # Apply transform if provided
        if self.transform is not None:
            data = self.transform(data)

        # Prepare output
        out = (data,)

        # Add target if specified
        if self.target is not None:
            if self.target_mapping is not None:
                out += (self.target_mapping[metadata[self.target]],)
            else:
                out += (np.float32(metadata[self.target]),)

        # Add metadata if requested
        if self.return_meta_data:
            out += (
                {
                    "participant_id": str(metadata["participant_id"]),
                    "session": str(metadata.get("session") or ""),
                    "run": str(metadata.get("run") or ""),
                    "segment_index": int(segment_idx),
                    "sample_index": int(idx),
                },
            )

        return out


class NpyEpochsDataset(BaseNpyDataset):
    def __init__(
        self,
        dataset_root: str | Path,
        dataset_index: str | Path | pd.DataFrame = None,
        target: str = None,
        target_mapping: dict = None,
        return_meta_data: bool = False,
        transform=None,
        cache: bool = False,
        scratch_root: str = None,
    ):
        """
        Dataset for pre-epoched EEG data (e.g., contrastChangeDetection task).
        Data is expected to be stored in .npy files of shape (n_epochs, n_channels,
        n_timepoints_per_epoch). Additionally, a labels file per recording can be
        provided with epoch-level labels.

        Args:
            dataset_root (str | Path):
                Root directory of the dataset.

            dataset_index (str | Path | pd.DataFrame, optional):
                Path to a CSV file or a DataFrame containing the following columns:
                    - 'participant_id': Unique identifier for each participant
                    - 'sfreq': Sampling frequency of the EEG recordings
                    - 'n_epochs': Number of epochs in the recording
                    - 'n_timepoints': Number of timepoints per epoch
                    - 'file': Path to the data file (NumPy .npy format) relative to dataset_root
                    - 'labels_file': Path to the labels file (structured NumPy .npy) relative to dataset_root
                    - If `target` is specified, target can come from either dataset_index or epoch labels.
                If None, will attempt to load dataset_index.csv from dataset_root.

            target (str, optional):
                Name of the column to use as target labels. Can be from dataset_index (recording-level)
                or from epoch labels (epoch-level). If specified, __getitem__ will return (data, target) tuples.

            target_mapping (dict, optional):
                Mapping from target values to numerical values. E.g. {"left": 0, "right": 1}.

            return_meta_data (bool, optional):
                If True, __getitem__ will additionally return information about the sample,
                such as participant_id and segment index within the recording.
                Returns a tuple of the form (data, metadata) or (data, target, metadata) if target is specified.

            transform (callable, optional):
                A callable (or composed transforms) to apply to data samples.
                Should accept and return a torch.Tensor. Use torchvision.transforms.Compose
                to combine multiple transforms.

            cache (bool, optional):
                If True, cache loaded files in memory for faster access. Defaults to False.

            scratch_root (str, optional):
                Path to a scratch directory. If provided, data will be saved to it when first loaded. If data
                is accessed in subsequent epochs, it will be loaded from the scratch directory instead of the
                original data directory. This is similar to caching, just that data is saved to a scratch directory
                instead of being kept in memory. Should be used when not enough memory is available for caching
                and data access is faster from the scratch directory than the original data directory.
        """
        super().__init__(dataset_root=dataset_root, cache=cache, scratch_root=scratch_root)

        self.target = target
        self.target_mapping = target_mapping
        self.return_meta_data = return_meta_data
        self.transform = transform

        # Load dataset index
        if dataset_index is None:
            self.dataset_index = pd.read_csv(self.dataset_root / "dataset_index.csv")
        elif isinstance(dataset_index, (str, Path)):
            self.dataset_index = pd.read_csv(dataset_index)
        elif isinstance(dataset_index, pd.DataFrame):
            self.dataset_index = dataset_index
        else:
            raise TypeError("If provided (not None), dataset_index must be a file path (str) or a pandas DataFrame.")

    @property
    def dataset_index(self):
        return self._dataset_index

    @dataset_index.setter
    def dataset_index(self, new_index):
        # Check that required columns are available
        required_cols = ["sfreq", "n_epochs", "n_timepoints", "file", "labels_file"]
        missing_cols = [col for col in required_cols if col not in new_index.columns]
        if missing_cols:
            raise ValueError(f"Dataset index missing required columns for epoched data: {missing_cols}")

        # Check that sfreq is consistent across all recordings
        if not np.all(new_index["sfreq"] == new_index["sfreq"].iloc[0]):
            raise ValueError("All recordings must have the same sampling frequency (sfreq).")

        # Drop rows with NaN values in target column if target is specified and exists in dataset_index
        if self.target is not None and self.target in new_index.columns:
            initial_length = len(new_index)
            new_index = new_index.dropna(subset=[self.target]).copy()
            dropped_count = initial_length - len(new_index)
            if dropped_count > 0:
                print(f"Warning: Dropped {dropped_count} recordings with NaN values in target column '{self.target}'")

            # Check if dataset is empty after filtering
            if len(new_index) == 0:
                raise ValueError("Dataset is empty after filtering. No valid recordings found.")

        self._dataset_index = new_index

        # Update dependent properties for epoched data
        self.cumulative_epochs = np.cumsum(self.dataset_index["n_epochs"].values)
        self.total_epochs = self.cumulative_epochs[-1]

    def __len__(self):
        """Return total number of epochs across all recordings."""
        return self.total_epochs

    def __getitem__(self, idx):
        """
        Returns:
            Returns an EEG epoch (and if specified during dataset construction, target and metadata).
            - data (torch.Tensor): EEG epoch of shape (n_channels, n_timepoints_per_epoch).
            - target (float | str | None): Target label if specified, else None.
            - metadata (dict): Metadata including participant_id, epoch-specific info, etc.
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self)} epochs")

        # Find which recording this epoch belongs to
        recording_idx = np.searchsorted(self.cumulative_epochs, idx, side="right")

        # Find epoch index within recording
        if recording_idx == 0:
            epoch_idx = idx
        else:
            epoch_idx = idx - self.cumulative_epochs[recording_idx - 1]

        # Get recording metadata
        metadata = self.dataset_index.iloc[recording_idx]

        # Load data
        data_file = self.dataset_root / metadata["file"]
        data = self._load_file(data_file)
        data = torch.tensor(data[epoch_idx, :, :])

        # Apply transform if provided
        if self.transform is not None:
            data = self.transform(data)

        # Prepare output
        out = (data,)

        # Add target if specified
        if self.target is not None:
            # Target can come from either recording metadata or epoch labels
            if self.target in metadata:
                target_value = metadata[self.target]
            else:
                if "labels_file" in metadata and pd.notna(metadata["labels_file"]):
                    # Load epoch labels from file
                    labels_file = self.dataset_root / metadata["labels_file"]
                    labels = self._load_file(labels_file)
                    epoch_labels = labels[epoch_idx]
                    if hasattr(epoch_labels.dtype, "names") and self.target in epoch_labels.dtype.names:
                        target_value = epoch_labels[self.target].item()
                    else:
                        raise ValueError(f"Target '{self.target}' not found in epoch labels or recording metadata")
                else:
                    raise ValueError(f"Target '{self.target}' not found in epoch labels or recording metadata")

            # Apply target mapping if provided
            if self.target_mapping is not None:
                target_value = self.target_mapping[target_value]
            else:
                target_value = np.float32(target_value)

            out += (target_value,)

        # Add metadata if requested
        if self.return_meta_data:
            out += (
                {
                    "participant_id": str(metadata["participant_id"]),
                    "session": str(metadata.get("session") or ""),
                    "run": str(metadata.get("run") or ""),
                    "segment_index": int(epoch_idx),
                    "sample_index": int(idx),
                },
            )

        return out


def subsample_dataset(
    dataset,
    n_participants: int | None = None,
    n_segments: int | None = None,
    segment_select: str = "random",
    unique_recording_per_participant: bool = True,
    stratify_by: str | None = None,
    seed: int = 42,
):
    """Subsample a dataset by number of participants and/or segments per recording.

    Produces a torch Subset of the original dataset containing only the selected
    segments. Either or both of n_participants and n_segments may be specified;
    if neither is set the dataset is returned unchanged.

    Args:
        dataset: An NpyDataset instance.
        n_participants: Number of participants to retain. If None, keep all.
        n_segments: Maximum segments per recording. If None, keep all.
        segment_select: How to choose segments within a recording.
            ``"random"`` — uniformly sample without replacement.
            ``"contiguous"`` — sample a contiguous block starting at a random offset.
        unique_recording_per_participant: When True (default), retain at most one
            randomly chosen recording per participant before any further subsampling.
        stratify_by: Column in dataset_index to stratify participant sampling by
            (e.g. the task target column). Only used when n_participants is set.
        seed: Random seed for reproducibility. With the same seed, a smaller
            n_participants or n_segments will always be a subset of a larger one —
            participants are selected from a fixed permutation and segments from a
            per-recording permutation derived from the seed.

    Returns:
        ``torch.utils.data.Subset`` with the selected indices, or the original
        dataset unchanged when no subsampling is requested.
    """
    if n_participants is None and n_segments is None:
        return dataset

    if segment_select not in ("random", "contiguous"):
        raise ValueError(f"segment_select must be 'random' or 'contiguous', got '{segment_select}'")

    rng = np.random.default_rng(seed)
    index = dataset.dataset_index.copy()
    index["_pos"] = np.arange(len(index))

    # Deduplicate to one recording per participant
    if unique_recording_per_participant and "participant_id" in index.columns:
        keep = []
        for _, group in index.groupby("participant_id"):
            keep.append(group.index[rng.integers(len(group))])
        index = index.loc[keep]

    # Subsample participants
    if n_participants is not None:
        participants = index["participant_id"].unique()
        if n_participants > len(participants):
            raise ValueError(
                f"Requested {n_participants} participants but only "
                f"{len(participants)} available in the dataset."
            )
        elif n_participants < len(participants):
            if stratify_by is not None and stratify_by in index.columns:
                labels = index.groupby("participant_id")[stratify_by].first()
                strata = sorted(labels.unique())
                base = n_participants // len(strata)
                remainder = n_participants % len(strata)
                selected = []
                for i, stratum in enumerate(strata):
                    pool = rng.permutation(labels[labels == stratum].index.values)
                    n_take = min(base + (1 if i < remainder else 0), len(pool))
                    selected.extend(pool[:n_take])
                if len(selected) < n_participants:
                    print(
                        f"Warning: stratified sampling yielded {len(selected)} participants "
                        f"instead of {n_participants} (a stratum has too few participants)."
                    )
                index = index[index["participant_id"].isin(selected)]
            else:
                index = index[index["participant_id"].isin(rng.permutation(participants)[:n_participants])]

    # Collect global segment indices — support both NpyDataset (cumulative_segments /
    # n_segments) and NpyEpochsDataset (cumulative_epochs / n_epochs).
    if hasattr(dataset, "cumulative_segments"):
        cumulative = dataset.cumulative_segments
        seg_col = "n_segments"
    else:
        cumulative = dataset.cumulative_epochs
        seg_col = "n_epochs"
    global_indices = []

    for _, row in index.iterrows():
        pos = row["_pos"]
        total_segs = row[seg_col]
        offset = int(cumulative[pos - 1]) if pos > 0 else 0

        if n_segments is not None and n_segments < total_segs:
            # Use a per-recording rng so segment ordering is independent of how
            # many participants were selected, ensuring nested subsets across configs.
            rec_rng = np.random.default_rng([seed, int(pos)])
            if segment_select == "random":
                local = np.sort(rec_rng.permutation(total_segs)[:n_segments])
            else:  # contiguous
                start = int(rec_rng.integers(0, total_segs - n_segments + 1))
                local = np.arange(start, start + n_segments)
        else:
            local = np.arange(total_segs)

        global_indices.extend(offset + local)

    global_indices = sorted(global_indices)

    print(
        f"Subsampled training data: {index['participant_id'].nunique()} participants, "
        f"{len(global_indices)}/{len(dataset)} segments "
        f"({100 * len(global_indices) / len(dataset):.1f}%)"
    )

    return Subset(dataset, global_indices)


def preprocess_bids_dataset(
    rawdata_root: str | Path,
    deriv_root: str | Path,
    channels: list[str],
    preprocessing: str,
    split_columns: dict[str, str] | None = None,
    dtype: str = "float32",
    sfreq: int = 100,
    num_worker: int = 1,
):
    # Get preprocessing settings
    preproc_settings = PREPROCESSING[preprocessing]

    # Find matching bids paths
    bids_root = Path(rawdata_root)
    bids_paths = find_matching_paths(
        root=bids_root, extensions=[".vhdr", ".edf", ".bdf", ".set"], datatypes="eeg", suffixes="eeg"
    )

    # Hot fix for potential duplicates due mne-bids bug: https://github.com/mne-tools/mne-bids/issues/1127
    bids_paths = list({bids_path.fpath: bids_path for bids_path in bids_paths}.values())

    results = Parallel(n_jobs=num_worker)(
        delayed(preprocess_bids_recording)(
            bids_path=bp,
            deriv_root=deriv_root,
            preproc_settings=preproc_settings,
            channels=channels,
            dtype=dtype,
            sfreq=sfreq,
        )
        for bp in tqdm(bids_paths, desc="Processing EEG files")
    )

    metadata = [r for r in results if r is not None]
    failed_files = [bids_paths[i] for i, r in enumerate(results) if r is None]

    # Create dataset index
    if not metadata:
        print("Preprocessing failed for all files.")
        return pd.DataFrame()

    metadata = pd.DataFrame(metadata)
    metadata.to_csv(Path(deriv_root) / "dataset_index.csv", index=False)
    print("Dataset index saved.")

    # Save split-specific index files
    if split_columns:
        for prefix, col in split_columns.items():
            if col not in metadata.columns:
                print(f"Warning: Split column '{col}' not found in metadata, skipping.")
                continue
            for split_name in ["train", "val", "test"]:
                filename = (
                    f"dataset_index_{split_name}.csv" if not prefix else f"dataset_index_{prefix}_{split_name}.csv"
                )
                split_df = metadata[metadata[col] == split_name]
                split_df.to_csv(Path(deriv_root) / filename, index=False)
                print(f"Dataset index for '{prefix or 'default'}/{split_name}' saved ({len(split_df)} recordings).")

    if failed_files:
        print(f"Failed to process {len(failed_files)} files:")
        for failed_file in failed_files:
            print(f"  {failed_file.basename}")

    return metadata


def preprocess_bids_recording(
    bids_path: BIDSPath,
    deriv_root: str | Path,
    channels: list[str],
    preproc_settings: dict,
    dtype: str = "float32",
    sfreq: int = 100,
):
    try:
        # Load and preprocess raw data
        raw = read_raw_bids(bids_path).load_data()
        raw = raw.pick(channels)
        if filter_settings := preproc_settings.get("filter"):
            raw.filter(l_freq=filter_settings.get("l_freq", None), h_freq=filter_settings.get("h_freq", None))
        if raw.info["sfreq"] != sfreq:
            raw.resample(sfreq)
        raw = raw.set_eeg_reference(ref_channels="average")

        # Construct output file path
        deriv_bids_path = bids_path.copy().update(root=deriv_root)
        deriv_bids_path.mkdir(exist_ok=True)
        outfile_path = deriv_bids_path.copy().update(extension=".npy", check=False)

        # Save scaled data in desired dtype
        data = (raw.get_data() * 1e5).astype(dtype)
        np.save(outfile_path.fpath, data)

        # Create metadata
        metadata = {"participant_id": bids_path.subject}
        recording_id = f"sub-{bids_path.subject}"
        if bids_path.session:
            metadata["session"] = bids_path.session
            recording_id += f"_ses-{bids_path.session}"
        if bids_path.run:
            metadata["run"] = bids_path.run
            recording_id += f"_run-{bids_path.run}"
        if bids_path.task:
            metadata["task"] = bids_path.task
            recording_id += f"_task-{bids_path.task}"
        metadata["n_timepoints"] = data.shape[-1]
        metadata["sfreq"] = sfreq
        metadata["file"] = str(outfile_path.fpath.relative_to(outfile_path.root))

        # Extract subject info from participants.tsv
        participants_tsv = pd.read_csv(bids_path.root / "participants.tsv", sep="\t")
        subject_info = participants_tsv[participants_tsv["participant_id"] == "sub-" + bids_path.subject].to_dict(
            "records"
        )[0]
        metadata.update(subject_info)

        # Extract recording info from scans.tsv
        if bids_path.session is not None:
            scants_tsv_path = (
                bids_path.root
                / f"sub-{bids_path.subject}"
                / f"ses-{bids_path.session}"
                / f"sub-{bids_path.subject}_ses-{bids_path.session}_scans.tsv"
            )
        else:
            scants_tsv_path = bids_path.root / f"sub-{bids_path.subject}" / f"sub-{bids_path.subject}_scans.tsv"

        if scants_tsv_path.exists():
            scans_tsv = pd.read_csv(scants_tsv_path, sep="\t")
            recording_info = scans_tsv[scans_tsv["filename"] == f"eeg/{bids_path.basename}"].to_dict("records")[0]
            metadata.update(recording_info)

        return metadata

    except Exception as e:
        print(f"Error processing {bids_path.fpath}: {str(e)}")
        return None
