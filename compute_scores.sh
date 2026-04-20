#!/bin/bash
#SBATCH --job-name=kronfluence_olmo2_32B_data_0
#SBATCH --account=a0107
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --output=logs/kronfluence_olmo2_32B_data_0_%j.out
#SBATCH --error=logs/kronfluence_olmo2_32B_data_0_%j.err

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

MODEL_SIZE="13B"
DATA_ID="data_0"
FACTORS_NAME="${MODEL_SIZE}-${DATA_ID}-factors"

torchrun --standalone --nnodes=1 --nproc-per-node=4 /iopsstor/scratch/cscs/laylaylo/IF-MultiLingual/olmo_kronfluence/compute_scores.py \
    --model_size $MODEL_SIZE \
    --data_id $DATA_ID \
    --factors_name $FACTORS_NAME \
    --scores_name raw \
    --factors_batch_size 4 \
    --query_gradient_rank 64

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