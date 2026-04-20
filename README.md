# Influence Analysis for OLMo-2 on CSCS

This pipeline is built on top of the **OpenWebText & Llama-3-8B example** from the [Kronfluence repository](https://github.com/pomonam/kronfluence/tree/main/examples/openwebtext). It has been modified to simplify running influence analysis on various model sizes (1B to 32B) within the CSCS environment.

This directory contains scripts for computing influence scores on the subset of Olmino-Mix dataset. 

## Changes and Additions

The following parameters have been added to automate model-specific configurations:

  * `--model_size`: Sets the model weights (1B, 7B, 13B, 32B) and automatically adjusts the layer count in `task.py`.
  * `--data_id`: Automatically maps to the correct prompt-completion file at `./data_{model_size}/{data_id}.json`.
  * `--covariance_model_partitions`: Sets module sharding for the covariance stage.
  * `--lambda_model_partitions`: Sets module sharding for the lambda stage.
  * `--data_partitions`: Sets data sharding for both stages.
  * `--output_dir`: Automatically routed to `/capstor/scratch/cscs/`.
---

## 🛠 Setup Guide

### 1. Install Dependencies
In your virtual environment, install all required packages:
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Open the `.sh` scripts and update the following environment variables to match your CSCS allocation:

* **`HF_HOME`**: Update this to a scratch directory (e.g., `/iopsstor/scratch/cscs/[user]/.cache/huggingface`) to store model weights. This prevents your home directory from exceeding its quota.
* **`VENV_PATH`**: Point this to the absolute path of your virtual environment.
* **`PYTHONPATH`**: The scripts `unset PYTHONPATH` to ensure a clean execution environment using only your local virtual environment.

### 3. Kronfluence Configuration 
Open `pipeline.py` to configure your specific models and pre-training data:

* **Training Dataset**: In the `get_olmino_dataset` function, locate the `load_dataset` call and replace `"laylaylo/olmino-mix-run1-15k"` with your preferred pre-training subset.
* **Model Mapping**: Update the `MODEL_MAP` dictionary to link your desired model sizes to their Hugging Face identifiers:
```python
MODEL_MAP = {
    "1B": "allenai/OLMo-2-0425-1B-Instruct",
    "7B": "allenai/OLMo-2-1124-7B-Instruct",
    "13B": "allenai/OLMo-2-1124-13B-Instruct",
    "32B": "allenai/OLMo-2-0325-32B-Instruct"
}
```

* **Prompt-Completion Data**: Ensure your query pairs are stored in the following directory structure: `./data/data_{model_size}/{data_id}.json` The pipeline is adjusted to find these files using the parameters passed in the shell scripts.

* **Output Directories**: In both `fit_factors.py` and `compute_scores.py`, locate the `full_output_dir` or `Analyzer` initialization and update the path with your username and preferred sub-folder structure to ensure results are stored in your specific CSCS scratch space.

---

## 🚀 Running the Analysis

### 1. Fitting Factors
Factors represent the model's curvature on the training data and only need to be computed **once per model size**.

* **Update Variables**: Set your desired `MODEL_SIZE`, `FACTORS_NAME` and sharding numbers (`COV_MOD`, `LAM_MOD`, etc.) at the top of the script.
* **Run**: `sbatch fit_factors.sh`

### 2. Computing Influence Scores
Scores are specific to your prompt-completion pairs. Run this for each `data_id` you wish to analyze.

* **Update Environment**: Ensure the `VENV_PATH` in `compute_scores.sh` matches your environment.
* **Update Variables**: Set the `MODEL_SIZE`, `DATA_ID`, and the **same** `FACTORS_NAME` used in Step 1.
* **Run**: `sbatch compute_scores.sh`

---

## 📋 Parameters
The shell scripts pass these parameters to the Python backend:
* `--model_size`: Automatically sets model weights and adjusts layer counts in `task.py` (16 to 64 layers).
* `--data_id`: Maps to the prompt-completion file at `./data/data_{model_size}/{data_id}.json`.
* `--covariance_model_partitions`: Shards modules during the covariance stage.
* `--lambda_model_partitions`: Shards modules during the lambda stage.
* `--data_partitions`: Shards the training dataset for memory efficiency.
* `--factor_batch_size`: Set to 4 by default; decrease to 1 or 2 if you encounter CUDA Out-of-Memory (OOM) errors.

## Credits

Original logic, `LanguageModelingTask` definitions, and EKFAC implementation are credited to the [Kronfluence](https://github.com/pomonam/kronfluence) team. This version adds the parameterization to handle deep architectures (up to 64 layers) and CSCS-specific file paths without manual code changes.