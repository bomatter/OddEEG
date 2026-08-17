import importlib
from pathlib import Path

import tyro

from oddeeg.user_config import user_config
from oddeeg.utils import resolve_scratch


def create_dataset(
    dataset_name: str,
    task: str | None = None,
    preprocessing: str = "minimal",
    dtype: str = "float32",
    segment_length: float = 2,  # in seconds
    sfreq: int = 100,  # sampling frequency
    transform=None,  # callable or composed transforms to apply to data samples (on the fly)
    rawdata_root: str | Path | None = None,  # will use user_config["data"][name]["rawdata_root"] if None
    deriv_root: str | Path | None = None,  # will use user_config["data"][name]["deriv_root"] if None
    data_split: str | None = None,  # name of the data split to use; if None, no splitting is performed
    pick: str | list[str] | None = None,  # can be used to select specific splits, e.g. ["train", "val"]
    num_worker: int = 1,  # number of workers for parallel dataset creation
    return_meta_data: list[str] | bool = False,  # True/False or list of split names for which to return meta data
    cache: bool = False,
    scratch_root=None,  # will use user_config["scratch_root"] if None
):
    """Factory function to create dataset instances."""

    if rawdata_root is None:
        rawdata_root = user_config["data"][dataset_name]["rawdata_root"]

    if deriv_root is None:
        deriv_root = user_config["data"][dataset_name]["deriv_root"]

    if scratch_root is None:
        scratch_root = resolve_scratch(scratch_root)

    try:
        dataset_module = importlib.import_module(f"oddeeg.datasets.{dataset_name.lower()}")
        return dataset_module.create_dataset(
            task=task,
            preprocessing=preprocessing,
            dtype=dtype,
            segment_length=segment_length,
            sfreq=sfreq,
            transform=transform,
            rawdata_root=rawdata_root,
            deriv_root=deriv_root,
            num_worker=num_worker,
            return_meta_data=return_meta_data,
            data_split=data_split,
            pick=pick,
            cache=cache,
            scratch_root=scratch_root,
        )
    except ModuleNotFoundError:
        raise ValueError(f"Dataset '{dataset_name}' is not available. Add a module in core/datasets/")


def get_dataset_info(name):
    """Get dataset-specific information like channels and dimensions.

    Args:
        name: Name of the dataset (e.g., 'TUAB')

    Returns:
        dict: Dictionary containing dataset information
    """
    try:
        dataset_module = importlib.import_module(f"oddeeg.datasets.{name.lower()}")
        info = {
            "n_channels": getattr(dataset_module, "N_CHANNELS", None),
            "channels": getattr(dataset_module, "CHANNELS", None),
        }

        return info
    except ModuleNotFoundError:
        raise ValueError(f"Dataset '{name}' is not available. Add a module in core/datasets/")


def get_supported_tasks(dataset_name):
    """Get the list of tasks supported by a specific dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'TUAB')

    Returns:
        list: List of supported task names
    """
    try:
        dataset_module = importlib.import_module(f"oddeeg.datasets.{dataset_name.lower()}")

        # Check if the module has SUPPORTED_TASKS defined
        if hasattr(dataset_module, "SUPPORTED_TASKS"):
            return list(dataset_module.SUPPORTED_TASKS.keys())
        else:
            return []
    except ModuleNotFoundError:
        raise ValueError(f"Dataset '{dataset_name}' is not available.")


def preprocess(
    dataset_name: str,  # name of the dataset (e.g., 'TUAB')
    task: str | None = None,  # task name
    preprocessing: str = "minimal",  # preprocessing pipeline name
    dtype: str = "float32",  # data type (e.g., 'float32')
    segment_length: float = 2.0,  # segment length in seconds
    sfreq: int = 100,  # sampling frequency in Hz
    rawdata_root: str | None = None,  # root directory for raw data
    deriv_root: str | None = None,  # root directory for data derivatives
    num_worker: int = 1,  # number of workers for parallel dataset creation
) -> None:
    """
    Create (preprocess) a dataset with specified parameters.

    Example usage (using entry point if package is installed):

        oddeeg-preprocess --dataset_name TUAB --preprocessing minimal

    With uv:

        uv run oddeeg-preprocess --dataset_name TUAB --preprocessing minimal

    Or as a module:

        python -m oddeeg.datasets.dataset_factory --dataset_name TUAB --preprocessing minimal
    """
    create_dataset(
        dataset_name=dataset_name,
        task=task,
        preprocessing=preprocessing,
        dtype=dtype,
        segment_length=segment_length,
        sfreq=sfreq,
        rawdata_root=rawdata_root,
        deriv_root=deriv_root,
        num_worker=num_worker,
        scratch_root=False,
    )


def main():
    tyro.cli(preprocess)


if __name__ == "__main__":
    main()
