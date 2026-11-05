#!/bin/bash
# Core features (G1-G8) for one plane per array task.
#   sbatch --array=0-45 slurm/extract_features.sbatch --training
#   sbatch --array=0-N  slurm/extract_features.sbatch --plane-list LIST.txt
#SBATCH -J cmclf_extract
#SBATCH -p batch
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /oscar/home/rtripp/data/rtripp/caiman_classifier_model/logs/%x_%A_%a.out
set -euo pipefail
REPO=${CMCLF_REPO:-/oscar/home/rtripp/data/rtripp/caiman_reader_RT}
PY=${CMCLF_PY:-$HOME/.conda/envs/caiman_classifier_env/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host $(hostname)  job ${SLURM_JOB_ID:-}  task ${SLURM_ARRAY_TASK_ID:-}  $(date)"
"$PY" "$REPO/classifier/scripts/extract_features.py" "$@"
