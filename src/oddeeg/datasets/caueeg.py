from pathlib import Path

from oddeeg.datasets.dataset import NpyDataset, preprocess_bids_dataset
from oddeeg.utils import construct_deriv_name

CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
]
N_CHANNELS = len(CHANNELS)

# Define supported tasks and their dataset-specific configurations
# to map from potentially different naming conventions to the standards
# defined in the tasks module.
SUPPORTED_TASKS = {
    "normal_mci_dementia": {
        "target_name": "dementia_label",
        "value_mapping": {"normal": 0, "mci": 1, "dementia": 2},
    },
    "age": {
        "target_name": "age",
        "value_mapping": None,
    },
}

# Mapping from data_split name to participants.tsv column used to create split index files.
# Preprocessing creates dataset_index_{split_name}_{train,val,test}.csv for each entry.
_SPLIT_COLUMNS = {
    "normality_no_overlap": "normality_split_no_overlap",
    "dementia_no_overlap": "dementia_split_no_overlap",
}


def create_dataset(
    rawdata_root: str | Path,
    deriv_root: str | Path,
    task: str | None = None,
    preprocessing: str = "minimal",
    dtype: str = "float32",
    segment_length: float = 2,  # in seconds
    sfreq: int = 100,  # sampling frequency
    transform=None,
    return_meta_data: list[str] | bool = False,
    data_split: str | None = None,
    pick=None,
    cache: bool = False,
    scratch_root=None,
    num_worker: int = 0,
):
    """Factory function to create CAUEEG dataset instances.

    Args:
        data_split: One of None (no split), 'default' (task-aware: 'normality_no_overlap' for the
            normality task, 'dementia_no_overlap' for all others), 'normality_no_overlap', or
            'dementia_no_overlap'.
    """
    if task is not None and task not in SUPPORTED_TASKS:
        supported = list(SUPPORTED_TASKS.keys())
        raise ValueError(f"Task '{task}' is not supported by the CAUEEG dataset. Supported tasks are: {supported}")

    deriv_name = construct_deriv_name(
        dataset_name="CAUEEG",
        preprocessing=preprocessing,
        dtype=dtype,
        sfreq=sfreq,
    )

    deriv_root = Path(deriv_root) / deriv_name
    if not deriv_root.exists() or not (deriv_root / "dataset_index.csv").exists():
        print(
            f"Preprocessing CAUEEG dataset with the following parameters:\n"
            f"preprocessing: {preprocessing}, dtype: {dtype}, sampling frequency: {sfreq} Hz\n"
            f"Preprocessed data will be saved at: {deriv_root}"
        )
        preprocess_bids_dataset(
            rawdata_root=rawdata_root,
            deriv_root=deriv_root,
            channels=CHANNELS,
            preprocessing=preprocessing,
            dtype=dtype,
            sfreq=sfreq,
            num_worker=num_worker,
            split_columns=_SPLIT_COLUMNS,
        )

    target_name = SUPPORTED_TASKS[task]["target_name"] if task is not None else None
    value_mapping = SUPPORTED_TASKS[task]["value_mapping"] if task is not None else None

    # Resolve "default" to the appropriate no-overlap split for this task
    if data_split == "default":
        effective_split = "normality_no_overlap" if task == "normality" else "dementia_no_overlap"
    else:
        effective_split = data_split

    if effective_split is None:  # load the entire dataset without splitting
        datasets = NpyDataset(
            dataset_root=deriv_root,
            segment_length=segment_length,
            transform=transform,
            target=target_name,
            target_mapping=value_mapping,
            return_meta_data=return_meta_data,
            cache=cache,
            scratch_root=Path(scratch_root) / deriv_name if scratch_root else scratch_root,
        )
    elif effective_split in _SPLIT_COLUMNS:
        datasets = {}
        for split_name in ["train", "val", "test"]:
            if pick is not None and split_name not in pick:
                continue

            datasets[split_name] = NpyDataset(
                dataset_root=deriv_root,
                dataset_index=deriv_root / f"dataset_index_{effective_split}_{split_name}.csv",
                segment_length=segment_length,
                transform=transform,
                target=target_name,
                target_mapping=value_mapping,
                return_meta_data=return_meta_data
                if isinstance(return_meta_data, bool)
                else (split_name in return_meta_data),
                cache=cache,
                scratch_root=Path(scratch_root) / deriv_name if scratch_root else scratch_root,
            )
    else:
        raise ValueError(
            f"Unsupported split option for CAUEEG: '{effective_split}'. "
            f"Valid options: None, 'default', {list(_SPLIT_COLUMNS.keys())}"
        )

    if effective_split is None or pick is None:
        return datasets
    elif isinstance(pick, str):
        return datasets[pick]
    elif isinstance(pick, list):
        if len(pick) == 1:
            return datasets[pick[0]]
        else:
            return tuple(datasets[p] for p in pick)
    else:
        raise TypeError("pick must be a string or a list of strings.")
