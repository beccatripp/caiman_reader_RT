"""Helpers shared by the scripts: login-node guard, SLURM array task, CPUs."""
import os
import socket
import sys


def guard_login_node():
    """OSCAR login nodes are shared: refuse to compute there. Fine inside any
    SLURM job (sbatch, interact) and on non-login hosts such as the VSCode node."""
    host = socket.gethostname()
    if host.startswith("login") and "SLURM_JOB_ID" not in os.environ:
        sys.exit(f"{host} is a login node: submit with sbatch (classifier/slurm/) or run in an "
                 f"`interact` session.")


def task_id(arg):
    if arg is not None:
        return arg
    v = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(v) if v is not None else None


def n_cpus(arg):
    if arg:
        return arg
    return int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
