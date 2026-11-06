#!/bin/bash
# xgboost-context stage 1: register all sessions of one mouse x plane per array task.
#   python scripts/register_fov.py --list          (groups and their index)
#   sbatch --array=0-11 slurm/register.sbatch [--run v1]
#SBATCH -J cmclf_register
#SBATCH -p batch
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:30:00
#SBATCH -o /oscar/home/rtripp/data/rtripp/caiman_classifier_model/logs/%x_%A_%a.out
set -euo pipefail
REPO=${CMCLF_REPO:-/oscar/home/rtripp/data/rtripp/caiman_reader_RT}
PY=${CMCLF_PY:-$HOME/.conda/envs/caiman_classifier_env/bin/python}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "host $(hostname)  job ${SLURM_JOB_ID:-}  task ${SLURM_ARRAY_TASK_ID:-}  $(date)"
"$PY" "$REPO/classifier/scripts/register_fov.py" "$@"
