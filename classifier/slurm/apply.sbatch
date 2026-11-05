#!/bin/bash
# Batch application: CHUNK planes per array task (default 5).
#   N=$(wc -l < LIST.txt); sbatch --array=0-$(( (N + 4) / 5 - 1 ))%20 slurm/apply.sbatch --version v20261105 --plane-list LIST.txt
#SBATCH -J cmclf_apply
#SBATCH -p batch
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /oscar/home/rtripp/data/rtripp/caiman_classifier_model/logs/%x_%A_%a.out
set -euo pipefail
REPO=${CMCLF_REPO:-/oscar/home/rtripp/data/rtripp/caiman_reader_RT}
PY=${CMCLF_PY:-$HOME/.conda/envs/caiman_classifier_env/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host $(hostname)  job ${SLURM_JOB_ID:-}  task ${SLURM_ARRAY_TASK_ID:-}  $(date)"
"$PY" "$REPO/classifier/scripts/apply.py" --chunk "${CHUNK:-5}" "$@"
