#!/bin/bash
# Optional G9 movie features (reads each plane's movie once).
#   sbatch --array=0-44 slurm/movie_features.sbatch --training
#SBATCH -J cmclf_movie
#SBATCH -p batch
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 01:30:00
#SBATCH -o /oscar/home/rtripp/data/rtripp/caiman_classifier_model/logs/%x_%A_%a.out
set -euo pipefail
REPO=${CMCLF_REPO:-/oscar/home/rtripp/data/rtripp/caiman_reader_RT}
PY=${CMCLF_PY:-$HOME/.conda/envs/caiman_classifier_env/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host $(hostname)  job ${SLURM_JOB_ID:-}  task ${SLURM_ARRAY_TASK_ID:-}  $(date)"
"$PY" "$REPO/classifier/scripts/extract_features.py" --movie "$@"
