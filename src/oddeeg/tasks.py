from enum import Enum

import torch.nn as nn


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


# Define task configurations
TASKS = {
    "normality": {
        "type": TaskType.CLASSIFICATION,
        "n_outputs": 2,
        "target_mapping": {0: "normal", 1: "abnormal"},
        "description": "Binary classification of normal vs abnormal EEG",
    },
    "normal_mci_dementia": {
        "type": TaskType.CLASSIFICATION,
        "n_outputs": 3,
        "target_mapping": {
            0: "normal",
            1: "mci",
            2: "dementia",
        },
        "description": "Multi-class classification of normal vs MCI vs dementia",
    },
    "age": {
        "type": TaskType.REGRESSION,
        "n_outputs": 1,
        "target_mapping": None,  # No mapping needed for regression
        "description": "Regression to predict age from EEG",
    },
    "sex": {
        "type": TaskType.CLASSIFICATION,
        "n_outputs": 2,
        "target_mapping": {0: "male", 1: "female"},
        "description": "Binary sex classification",
    },
    "left_right_hand": {
        "type": TaskType.CLASSIFICATION,
        "n_outputs": 2,
        "target_mapping": {0: "left_hand", 1: "right_hand"},
        "description": "Binary motor imagery classification: left hand vs right hand",
    },
}


## Helper functions
#


def get_task_info(task):
    return TASKS[task]


def get_metrics(task):
    if task not in TASKS:
        raise ValueError(f"Task '{task}' is not defined.")

    if TASKS[task]["type"] == TaskType.REGRESSION:
        return {
            "MeanSquaredError": {},
            "MeanAbsoluteError": {},
        }
    elif TASKS[task]["type"] == TaskType.CLASSIFICATION:
        return {
            "MulticlassAccuracy": {"num_classes": TASKS[task]["n_outputs"]},
            "MulticlassPrecision": {"num_classes": TASKS[task]["n_outputs"]},
            "MulticlassRecall": {"num_classes": TASKS[task]["n_outputs"]},
            "MulticlassF1Score": {"num_classes": TASKS[task]["n_outputs"]},
            "MulticlassAUROC": {"num_classes": TASKS[task]["n_outputs"]},
            "MulticlassAveragePrecision": {"num_classes": TASKS[task]["n_outputs"], "validate_args": False},  # AUPRC
        }
    else:
        raise ValueError(f"Task type '{TASKS[task]['type']}' is not supported.")


def get_loss_function(task):
    if task not in TASKS:
        raise ValueError(f"Task '{task}' is not defined.")

    if TASKS[task]["type"] == TaskType.REGRESSION:
        return nn.MSELoss()
    elif TASKS[task]["type"] == TaskType.CLASSIFICATION:
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Task type '{TASKS[task]['type']}' is not supported.")


def get_target_dtype(task):
    if task not in TASKS:
        raise ValueError(f"Task '{task}' is not defined.")

    if TASKS[task]["type"] == TaskType.REGRESSION:
        return "float32"
    elif TASKS[task]["type"] == TaskType.CLASSIFICATION:
        return "int64"
    else:
        raise ValueError(f"Task type '{TASKS[task]['type']}' is not supported.")


def get_target_mapping(task):
    if task not in TASKS:
        raise ValueError(f"Task '{task}' is not defined.")

    return TASKS[task]["target_mapping"]
