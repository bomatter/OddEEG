# OddEEG: Out-of-distribution Detection for EEG

This repository contains the research code for the experiments in our paper *"OOD Detection for EEG-based Machine Learning in High-Risk Environments"*.



## Install & Setup

1. Install dependencies with [uv](https://docs.astral.sh/uv/):

   ```bash
   uv sync
   ```

2. Request access to the datasets and refer to the following repositories to harmonise and convert them to BIDS format.

   - TUAB: https://github.com/bomatter/data-TUAB
   - CAUEEG: https://github.com/bomatter/data-CAUEEG
   - BNCI: https://ww2.nemar.org/dataset/nm000139 (direct download, no request or conversion required)

3. Create a copy of the `user_config.example.yaml` file and rename it to `user_config.yaml`. Then open it and configure the paths to the dataset folders and the directory, where you want outputs to be saved.



## Preprocess Data

Example usage:

```bash
uv run oddeeg-preprocess --dataset_name TUAB --num_worker 32
```



## Model Training

Example usage:

```bash
uv run oddeeg-train \
    --dataset_name TUAB \
    --task normality \
    --model_name TCN \
    --training_mode discriminative
```

Run `uv run oddeeg-train --help` for the full list of options.



## Model Evaluation

Example usage:

```bash
uv run oddeeg-eval \
    --config path/to/your/training_output/config.yml \
    --perturbation_sfreq 125
```

Run `uv run oddeeg-eval --help` for the full list of options.



## Reproducing Paper Results

Refer to the notebooks in the `experiments/` folder for guidance on how to reproduce the main results of the paper. Training and evaluation jobs can be conveniently submitted as slurm jobs from these notebooks or alternatively executed manually via the CLI as described above and in the notebooks.

Some OOD detection methods require a further fitting or calibration step after statistics are extracted in oddeeg-eval jobs. Details for this are also provided in the notebooks.