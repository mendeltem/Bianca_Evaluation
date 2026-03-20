#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SLURM Launcher for cluster_size_grid_search.py (Array Mode)
=============================================================
@author: temuuleu

Submits:
  1. Array job (--array=0-4): each task processes one fold
  2. Merge job (--dependency=afterok:<array_id>): merges checkpoints
"""

import os
import re
import subprocess
from dotenv import load_dotenv

load_dotenv()

# =============================================================
# CONFIG
# =============================================================
SLURM_ACCOUNT   = os.getenv("SLURM_ACCOUNT", "sc-users")
SLURM_MAIL_USER = os.getenv("SLURM_MAIL_USER", "")
PARTITION_TYPE   = os.getenv("SLURM_PARTITION_TYPE", "cpu")
SLURM_CPUS      = os.getenv("SLURM_CPUS", "32")
SLURM_MEM       = os.getenv("SLURM_MEM", "64G")
SLURM_TIME      = os.getenv("SLURM_TIME", "48:00:00")
CONDA_ENV       = os.getenv("CONDA_ENV",
                             "/home/temuuleu/SC_Stroke_MRI/Software/miniconda3/envs/bianca_env")
MINIFORGE_PATH  = os.getenv("MINIFORGE_PATH",
                             "/opt/miniforge/etc/profile.d/conda.sh")

# =============================================================
# PATHS
# =============================================================
WORK_DIR    = os.getcwd()
CMD         = os.path.join(WORK_DIR, "4_cluster_size_grid_search.py")
LOGS_DIR    = os.path.join(WORK_DIR, "log")
JOB_NAME    = "clust_gs"

os.makedirs(LOGS_DIR, exist_ok=True)

# =============================================================
# PARTITION
# =============================================================
if PARTITION_TYPE == "gpu":
    partition_block = "#SBATCH --partition=gpu\n#SBATCH --gres=shard:1"
else:
    partition_block = ""

# =============================================================
# CONDA BLOCK (shared)
# =============================================================
conda_block = f"""
if [ -f {MINIFORGE_PATH} ]; then
    source {MINIFORGE_PATH}
else
    echo "ERROR: Miniforge not found at {MINIFORGE_PATH}"
    exit 1
fi
conda activate {CONDA_ENV}
export MPLBACKEND=Agg
cd {WORK_DIR}
"""

# =============================================================
# 1) ARRAY JOB SCRIPT (fold per task)
# =============================================================
array_script = f"""#!/bin/bash
#SBATCH --job-name={JOB_NAME}
#SBATCH --array=0-4
{partition_block}
#SBATCH --account={SLURM_ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={SLURM_CPUS}
#SBATCH --mem={SLURM_MEM}
#SBATCH --time={SLURM_TIME}
#SBATCH --output={LOGS_DIR}/{JOB_NAME}_fold%a_%j.out
#SBATCH --error={LOGS_DIR}/{JOB_NAME}_fold%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={SLURM_MAIL_USER}

echo "Fold $SLURM_ARRAY_TASK_ID | Job $SLURM_JOB_ID | Host $(hostname) | CPUs $SLURM_CPUS_PER_TASK"
{conda_block}
python {CMD}

echo "Fold $SLURM_ARRAY_TASK_ID completed."
"""

# =============================================================
# 2) MERGE JOB SCRIPT (lightweight, runs after all folds)
# =============================================================
merge_script = f"""#!/bin/bash
#SBATCH --job-name={JOB_NAME}_merge
#SBATCH --account={SLURM_ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output={LOGS_DIR}/{JOB_NAME}_merge_%j.out
#SBATCH --error={LOGS_DIR}/{JOB_NAME}_merge_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={SLURM_MAIL_USER}

echo "Merge job $SLURM_JOB_ID on $(hostname)"
{conda_block}
python {CMD} --merge

echo "Merge completed."
"""

# =============================================================
# WRITE & SUBMIT
# =============================================================
array_file = os.path.join(WORK_DIR, f"{JOB_NAME}_array.sh")
merge_file = os.path.join(WORK_DIR, f"{JOB_NAME}_merge.sh")

with open(array_file, "w") as f:
    f.write(array_script)
os.chmod(array_file, 0o755)

with open(merge_file, "w") as f:
    f.write(merge_script)
os.chmod(merge_file, 0o755)

print(f"Work dir:  {WORK_DIR}")
print(f"CPUs/fold: {SLURM_CPUS}")
print(f"Memory:    {SLURM_MEM}")
print(f"Time:      {SLURM_TIME}")

# --- Submit array job ---
print(f"\nSubmitting array job (5 folds)...")
r1 = subprocess.run(["sbatch", array_file], capture_output=True, text=True)

if r1.returncode != 0:
    print(f"Error: {r1.stderr.strip()}")
    exit(1)

print(f"  {r1.stdout.strip()}")

# Extract array job ID
match = re.search(r"(\d+)", r1.stdout)
if not match:
    print("Could not parse array job ID, submit merge manually:")
    print(f"  sbatch --dependency=afterok:<ARRAY_JOB_ID> {merge_file}")
    exit(0)

array_job_id = match.group(1)

# --- Submit merge job with dependency ---
print(f"Submitting merge job (after {array_job_id})...")
r2 = subprocess.run(
    ["sbatch", f"--dependency=afterok:{array_job_id}", merge_file],
    capture_output=True, text=True,
)

if r2.returncode == 0:
    print(f"  {r2.stdout.strip()}")
    print(f"\nDone. Monitor with: squeue -u $USER")
else:
    print(f"  Error: {r2.stderr.strip()}")
    print(f"  Submit manually: sbatch --dependency=afterok:{array_job_id} {merge_file}")