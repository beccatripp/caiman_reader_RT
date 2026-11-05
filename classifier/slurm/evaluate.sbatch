#!/bin/bash
#   sbatch slurm/evaluate.sbatch --version v20261105 --locked-test
#   sbatch slurm/evaluate.sbatch --version v20261105 --compare --plane-list LIST.txt
#SBATCH -J cmclf_eval
#SBATCH -p batch
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /oscar/home/rtripp/data/rtripp/caiman_classifier_model/logs/%x_%j.out
set -euo pipefail
REPO=${CMCLF_REPO:-/oscar/home/rtripp/data/rtripp/caiman_reader_RT}
PY=${CMCLF_PY:-$HOME/.conda/envs/caiman_classifier_env/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host $(hostname)  job ${SLURM_JOB_ID:-}  task ${SLURM_ARRAY_TASK_ID:-}  $(date)"
"$PY" "$REPO/classifier/scripts/evaluate.py" "$@"
