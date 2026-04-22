#!/bin/bash
#SBATCH --job-name=if_olmo_32B_4_8_12h
#SBATCH --account=a0107
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --output=logs/if_olmo_32B_4_8_12h_%j.out
#SBATCH --error=logs/if_olmo_32B_4_8_12h_%j.err

#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

# --- ENVIRONMENT SETUP ---
unset PYTHONPATH
export HF_HOME=/iopsstor/scratch/cscs/laylaylo/.cache/huggingface 

VENV_PATH=/iopsstor/scratch/cscs/laylaylo/IF-MultiLingual/.venv_clariden
source $VENV_PATH/bin/activate
PYTHON_BIN=$VENV_PATH/bin/python

# Start timing
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
START_EPOCH=$(date +%s)
echo "Job started at: $START_TIME" | tee -a logs/timing.log

# --- KRONFLUENCE SETUP ---

# # 1B - took 35min
# MODEL_SIZE="1B"
# COV_MOD=1
# LAM_MOD=1
# DATA_PART=1
# BATCH_SIZE=4

# # 7B - took 2h30min
# MODEL_SIZE="7B"
# COV_MOD=1
# LAM_MOD=2
# DATA_PART=1
# BATCH_SIZE=4

# # 13B - takes 4h30min
# MODEL_SIZE="13B"
# COV_MOD=2
# LAM_MOD=4
# DATA_PART=1
# BATCH_SIZE=4

# 32B - takes
MODEL_SIZE="32B"
COV_MOD=4
LAM_MOD=8
DATA_PART=1
BATCH_SIZE=2

FACTORS_NAME="exp1-${MODEL_SIZE}-factors-cov_part_${COV_MOD}-lambda_part${LAM_MOD}"

torchrun --standalone --nnodes=1 --nproc-per-node=4 /iopsstor/scratch/cscs/laylaylo/IF-MultiLingual/kronfluence_olmo/fit_factors.py \
    --model_size $MODEL_SIZE \
    --factors_name $FACTORS_NAME \
    --covariance_model_partitions $COV_MOD \
    --lambda_model_partitions $LAM_MOD \
    --data_partitions $DATA_PART \
    --factor_batch_size $BATCH_SIZE # decrease if OOM

# End timing
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
END_EPOCH=$(date +%s)
DURATION=$((END_EPOCH - START_EPOCH))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "Job ended at: $END_TIME" | tee -a logs/timing.log
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s" | tee -a logs/timing.log
echo "Job completed successfully"