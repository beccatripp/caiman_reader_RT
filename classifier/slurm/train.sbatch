#!/bin/bash
# Nested CV, leave-one-animal-out, ablation, final dev + deploy models.
#   sbatch slurm/train.sbatch --version v20261105
#   sbatch slurm/train.sbatch --version v20261105 --stages final
#SBATCH -J cmclf_train
#SBATCH -p batch
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o /oscar/home/rtripp/data/rtripp/caiman_classifier_model/logs/%x_%j.out
set -euo pipefail
REPO=${CMCLF_REPO:-/oscar/home/rtripp/data/rtripp/caiman_reader_RT}
PY=${CMCLF_PY:-$HOME/.conda/envs/caiman_classifier_env/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host $(hostname)  job ${SLURM_JOB_ID:-}  task ${SLURM_ARRAY_TASK_ID:-}  $(date)"
"$PY" "$REPO/classifier/scripts/train.py" "$@"
