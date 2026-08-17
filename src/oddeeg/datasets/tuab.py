from pathlib import Path

from oddeeg.datasets.dataset import NpyDataset, preprocess_bids_dataset
from oddeeg.utils import construct_deriv_name

# Dataset-specific constants
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
    "normality": {"target_name": "normality", "value_mapping": {"normal": 0, "abnormal": 1}},
    "sex": {"target_name": "sex", "value_mapping": {"M": 0, "F": 1}},
    "age": {"target_name": "age", "value_mapping": None},
}

# participants.tsv column that encodes the train/val/test split
_SPLIT_COLUMNS = {"": "train_val_test_split"}


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
    num_worker=0,  # number of workers for parallel dataset creation
):
    # Check if the requested target is supported
    if task is not None and task not in SUPPORTED_TASKS:
        supported = list(SUPPORTED_TASKS.keys())
        raise ValueError(f"Task '{task}' is not supported by the TUAB dataset. Supported tasks are: {supported}")

    deriv_name = construct_deriv_name(
        dataset_name="TUAB",
        preprocessing=preprocessing,
        dtype=dtype,
        sfreq=sfreq,
    )

    # Check if requested dataset has already been preprocessed
    deriv_root = Path(deriv_root) / deriv_name
    if not deriv_root.exists() or not (deriv_root / "dataset_index.csv").exists():
        print(
            f"Preprocessing TUAB dataset with the following parameters:\n"
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

    if data_split is None:  # load the entire dataset without splitting
        datasets = NpyDataset(
            dataset_root=deriv_root,
            segment_length=segment_length,
            transform=transform,
            target=SUPPORTED_TASKS[task]["target_name"] if task is not None else None,
            target_mapping=SUPPORTED_TASKS[task]["value_mapping"] if task is not None else None,
            return_meta_data=return_meta_data,
            cache=cache,
            scratch_root=Path(scratch_root) / deriv_name if scratch_root else scratch_root,
        )
    elif data_split == "default":  # use predefined train/val/test split
        datasets = {}
        for split_name in ["train", "val", "test"]:
            if pick is not None and split_name not in pick:
                continue

            datasets[split_name] = NpyDataset(
                dataset_root=deriv_root,
                dataset_index=deriv_root / f"dataset_index_{split_name}.csv",
                segment_length=segment_length,
                transform=transform,
                target=SUPPORTED_TASKS[task]["target_name"] if task is not None else None,
                target_mapping=SUPPORTED_TASKS[task]["value_mapping"] if task is not None else None,
                return_meta_data=return_meta_data
                if isinstance(return_meta_data, bool)
                else (split_name in return_meta_data),
                cache=cache,
                scratch_root=Path(scratch_root) / deriv_name if scratch_root else scratch_root,
            )
    else:
        raise ValueError(f"Unsupported split option: {data_split}")

    if pick is None:
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
