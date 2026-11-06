#!/bin/bash
# xgboost-context stage 1: register the sessions of one mouse x plane per array task
# (each to its next 3 sessions, and all to one reference). About 20-30 min per group.
#   python scripts/register_fov.py --list          (groups and their index)
#   sbatch --array=0-11 slurm/register.sbatch [--run v2]
#SBATCH -J cmclf_register
#SBATCH -p batch
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -o /oscar/home/rtripp/data/rtripp/caiman_classifier_model/logs/%x_%A_%a.out
set -euo pipefail
REPO=${CMCLF_REPO:-/oscar/home/rtripp/data/rtripp/caiman_reader_RT}
PY=${CMCLF_PY:-$HOME/.conda/envs/caiman_classifier_env/bin/python}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "host $(hostname)  job ${SLURM_JOB_ID:-}  task ${SLURM_ARRAY_TASK_ID:-}  $(date)"
"$PY" "$REPO/classifier/scripts/register_fov.py" "$@"
