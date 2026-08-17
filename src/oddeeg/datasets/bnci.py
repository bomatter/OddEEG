from pathlib import Path

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mne_bids import BIDSPath, find_matching_paths, read_raw_bids
from tqdm import tqdm

from oddeeg.datasets.dataset import NpyEpochsDataset, PREPROCESSING
from oddeeg.utils import construct_deriv_name


CHANNELS = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
]
N_CHANNELS = len(CHANNELS)

# Supported tasks and how they map BNCI trial types to target integers
SUPPORTED_TASKS = {
    "left_right_hand": {
        "target_name": "trial_type",
        "value_mapping": {"left_hand": 0, "right_hand": 1},
    },
}

# Run index (0-based) held out for validation within each participant's training session.
_VAL_RUN = 5


def create_dataset(
    rawdata_root: str | Path,
    deriv_root: str | Path,
    task: str | None = None,
    preprocessing: str = "minimal",
    dtype: str = "float32",
    segment_length: float = 4.0,
    sfreq: int = 100,
    transform=None,
    return_meta_data: list[str] | bool = False,
    data_split: str | None = None,
    pick=None,
    cache: bool = False,
    scratch_root=None,
    num_worker: int = 1,
):
    """Factory function to create BNCI 2014-001 dataset instances.

    Args:
        data_split: Controls which participant is used for training.
            - None: Load all recordings without splitting.
            - "default" or "1"-"9": Use the given participant's ses-0train data for
              training (run 0-4) and validation (run 5). All participants' ses-1test
              data form the test set. "default" is an alias for "1".
    """
    if task is not None and task not in SUPPORTED_TASKS:
        supported = list(SUPPORTED_TASKS.keys())
        raise ValueError(f"Task '{task}' is not supported by the BNCI dataset. Supported tasks: {supported}")

    deriv_name = construct_deriv_name(
        dataset_name="BNCI",
        preprocessing=preprocessing,
        dtype=dtype,
        segment_length=segment_length,
        sfreq=sfreq,
    )

    deriv_root = Path(deriv_root) / deriv_name
    if not deriv_root.exists() or not (deriv_root / "dataset_index.csv").exists():
        print(
            f"Preprocessing BNCI dataset with the following parameters:\n"
            f"preprocessing: {preprocessing}, dtype: {dtype}, sampling frequency: {sfreq} Hz\n"
            f"Preprocessed data will be saved at: {deriv_root}"
        )
        preprocess_bnci_dataset(
            rawdata_root=rawdata_root,
            deriv_root=deriv_root,
            preprocessing=preprocessing,
            dtype=dtype,
            segment_length=segment_length,
            sfreq=sfreq,
            num_worker=num_worker,
        )

    task_cfg = SUPPORTED_TASKS[task] if task is not None else None
    target_name = task_cfg["target_name"] if task_cfg is not None else None
    value_mapping = task_cfg["value_mapping"] if task_cfg is not None else None

    scratch = Path(scratch_root) / deriv_name if scratch_root else scratch_root

    def _make_dataset(index_path, return_meta):
        return NpyEpochsDataset(
            dataset_root=deriv_root,
            dataset_index=index_path,
            target=target_name,
            target_mapping=value_mapping,
            return_meta_data=return_meta,
            transform=transform,
            cache=cache,
            scratch_root=scratch,
        )

    def _return_meta_for(split_name):
        if isinstance(return_meta_data, bool):
            return return_meta_data
        return split_name in return_meta_data

    if data_split is None:
        return _make_dataset(deriv_root / "dataset_index.csv", return_meta_data)

    if data_split == "default":
        data_split = "1"

    try:
        train_participant = int(data_split)
    except ValueError:
        raise ValueError(
            f"Unsupported data_split for BNCI: '{data_split}'. "
            "Use 'default', or '1'-'9' to select a training participant."
        )
    if train_participant < 1 or train_participant > 9:
        raise ValueError(f"BNCI has participants 1-9, got data_split='{data_split}'.")

    datasets = {}
    split_files = {
        "train": deriv_root / f"dataset_index_participant-{train_participant}_train.csv",
        "val": deriv_root / f"dataset_index_participant-{train_participant}_val.csv",
        "test": deriv_root / "dataset_index_test.csv",
    }

    for split_name, index_path in split_files.items():
        if pick is not None and split_name not in pick:
            continue
        if not index_path.exists():
            raise FileNotFoundError(
                f"Split index file not found: {index_path}. "
                "Re-run preprocessing to regenerate split index files."
            )
        datasets[split_name] = _make_dataset(index_path, _return_meta_for(split_name))

    if pick is None:
        return datasets
    elif isinstance(pick, str):
        return datasets[pick]
    elif isinstance(pick, list):
        if len(pick) == 1:
            return datasets[pick[0]]
        return tuple(datasets[p] for p in pick)
    else:
        raise TypeError("pick must be a string or a list of strings.")



def preprocess_bnci_dataset(
    rawdata_root: str | Path,
    deriv_root: str | Path,
    preprocessing: str = "minimal",
    dtype: str = "float32",
    segment_length: float = 4.0,
    sfreq: int = 100,
    num_worker: int = 1,
):
    """Epoch and preprocess all BNCI recordings, then write split index files."""
    preproc_settings = PREPROCESSING[preprocessing]
    rawdata_root = Path(rawdata_root)
    deriv_root = Path(deriv_root)
    deriv_root.mkdir(parents=True, exist_ok=True)

    bids_paths = find_matching_paths(
        root=rawdata_root,
        extensions=[".bdf"],
        datatypes="eeg",
        suffixes="eeg",
    )
    bids_paths = list({bp.fpath: bp for bp in bids_paths}.values())

    results = Parallel(n_jobs=num_worker)(
        delayed(_preprocess_bnci_recording)(bp, deriv_root, preproc_settings, dtype, segment_length, sfreq)
        for bp in tqdm(bids_paths, desc="Processing BNCI recordings")
    )

    records = [r for r in results if r is not None]
    if not records:
        raise RuntimeError("Preprocessing failed for all BNCI recordings.")

    metadata = pd.DataFrame(records)
    metadata.to_csv(deriv_root / "dataset_index.csv", index=False)
    print("BNCI dataset index saved.")

    _write_split_indices(metadata, deriv_root)


def _preprocess_bnci_recording(
    bids_path: BIDSPath,
    deriv_root: Path,
    preproc_settings: dict,
    dtype: str,
    segment_length: float,
    sfreq: int,
):
    try:
        raw = read_raw_bids(bids_path, verbose=False).load_data(verbose=False)
        raw.pick(CHANNELS)

        if filter_settings := preproc_settings.get("filter"):
            raw.filter(
                l_freq=filter_settings.get("l_freq", None),
                h_freq=filter_settings.get("h_freq", None),
                verbose=False,
            )
        if raw.info["sfreq"] != sfreq:
            raw.resample(sfreq, verbose=False)
        raw.set_eeg_reference(ref_channels="average", verbose=False)

        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # Epoch around left/right hand motor imagery classes
        epochs = mne.Epochs(
            raw,
            events,
            event_id={'left_hand': event_id['left_hand'], 'right_hand': event_id['right_hand']},
            tmin=0.0,
            tmax=segment_length - 1 / sfreq,
            baseline=None,
            preload=True,
            verbose=False,
        )

        if len(epochs) == 0:
            print(f"No valid epochs found in {bids_path.fpath}, skipping.")
            return None

        data = (epochs.get_data(copy=False) * 1e5).astype(dtype)
        labels = epochs.events[:, 2]
        label_names = np.array([list(event_id.keys())[list(event_id.values()).index(c)] for c in labels])

        # Construct output paths mirroring the BIDS structure
        subject = bids_path.subject
        session = bids_path.session
        run = bids_path.run

        out_dir = deriv_root / f"sub-{subject}" / f"ses-{session}" / "eeg"
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = f"sub-{subject}_ses-{session}_task-imagery_run-{run}"
        data_file = out_dir / f"{stem}_eeg.npy"
        labels_file = out_dir / f"{stem}_labels.npy"

        np.save(data_file, data)
        structured_labels = np.array(
            [(name,) for name in label_names],
            dtype=[("trial_type", "U20")],
        )
        np.save(labels_file, structured_labels)

        record = {
            "participant_id": subject,
            "session": session,
            "run": run,
            "sfreq": sfreq,
            "n_epochs": data.shape[0],
            "n_timepoints": data.shape[2],
            "file": str(data_file.relative_to(deriv_root)),
            "labels_file": str(labels_file.relative_to(deriv_root)),
        }

        # Merge participant metadata from participants.tsv
        participants_tsv = pd.read_csv(bids_path.root / "participants.tsv", sep="\t")
        subject_row = participants_tsv[participants_tsv["participant_id"] == f"sub-{subject}"]
        if not subject_row.empty:
            for col in ["age", "sex", "hand"]:
                if col in subject_row.columns:
                    record[col] = subject_row.iloc[0][col]

        return record

    except Exception as e:
        print(f"Error processing {bids_path.fpath}: {e}")
        return None


def _write_split_indices(metadata: pd.DataFrame, deriv_root: Path):
    """Write per-participant train/val index files and a shared test index file."""
    train_session = "0train"
    test_session = "1test"

    test_df = metadata[metadata["session"] == test_session].copy()
    test_df.to_csv(deriv_root / "dataset_index_test.csv", index=False)
    print(f"Test index saved ({len(test_df)} recordings).")

    participants = metadata["participant_id"].unique()
    for participant in participants:
        participant_train = metadata[
            (metadata["participant_id"] == participant) & (metadata["session"] == train_session)
        ].copy()

        # Sort by run so the held-out run is always the last one
        participant_train = participant_train.sort_values("run").reset_index(drop=True)

        val_mask = participant_train["run"] == str(_VAL_RUN)
        train_df = participant_train[~val_mask]
        val_df = participant_train[val_mask]

        n = int(participant)
        train_df.to_csv(deriv_root / f"dataset_index_participant-{n}_train.csv", index=False)
        val_df.to_csv(deriv_root / f"dataset_index_participant-{n}_val.csv", index=False)
        print(
            f"Participant {n}: train={len(train_df)} recordings, val={len(val_df)} recordings."
        )
