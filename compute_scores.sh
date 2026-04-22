#!/bin/bash
#SBATCH --job-name=if_olmo_1B_score_data_34
#SBATCH --account=a0107
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=450G
#SBATCH --time=02:30:00
#SBATCH --output=logs/if_olmo_1B_score_data_34_%j.out
#SBATCH --error=logs/if_olmo_1B_score_data_34_%j.err

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

# 1B - took 1h20min
MODEL_SIZE="1B"
COV_MOD=1
LAM_MOD=1
BATCH_SIZE=16

# # 7B -  
# MODEL_SIZE="7B"
# COV_MOD=1
# LAM_MOD=2
# BATCH_SIZE=4 

# # 13B - 
# MODEL_SIZE="13B"
# COV_MOD=2
# LAM_MOD=4
# BATCH_SIZE=4 

FACTORS_NAME="exp1-${MODEL_SIZE}-factors-cov_part_${COV_MOD}-lambda_part${LAM_MOD}"

DATA_ID="data_34"
SCORES_NAME="${MODEL_SIZE}-${DATA_ID}-raw"

torchrun --standalone --nnodes=1 --nproc-per-node=4 /iopsstor/scratch/cscs/laylaylo/IF-MultiLingual/kronfluence_olmo/compute_scores.py \
    --model_size $MODEL_SIZE \
    --data_id $DATA_ID \
    --factors_name $FACTORS_NAME \
    --scores_name $SCORES_NAME \
    --train_batch_size $BATCH_SIZE \
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