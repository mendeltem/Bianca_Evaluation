#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SLURM Runner: Threshold Analysis All Seeds
===========================================
Submits one SLURM job per seed (array job 1-10).
Each job runs 3_threshold_analysis_all_seeds.py with SLURM_SEED=<seed>.
"""

import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

SLURM_ACCOUNT    = os.getenv("SLURM_ACCOUNT", "sc-users")
SLURM_MAIL_USER  = os.getenv("SLURM_MAIL_USER")
PARTITION_TYPE    = os.getenv("SLURM_PARTITION_TYPE", "cpu")
SLURM_CPUS       = os.getenv("SLURM_CPUS", "8")
SLURM_MEM        = os.getenv("SLURM_MEM", "64G")
SLURM_TIME       = os.getenv("SLURM_TIME", "24:00:00")
CONDA_ENV        = os.getenv("CONDA_ENV", "/home/temuuleu/SC_Stroke_MRI/Software/miniconda3/envs/bianca_env")
MINIFORGE_PATH   = os.getenv("MINIFORGE_PATH", "/opt/miniforge/etc/profile.d/conda.sh")

current_dir = os.getcwd()
cmd = os.path.join(current_dir, "3_threshold_analysis_all_seeds.py")
logs_dir = os.path.join(os.path.dirname(cmd), "log")
os.makedirs(logs_dir, exist_ok=True)

job_name = "th_all_seeds"

if PARTITION_TYPE == "gpu":
    partition = """#SBATCH --partition=gpu
#SBATCH --gres=shard:1"""
else:
    partition = ""

slurm_script = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --array=1-10
{partition}
#SBATCH --account={SLURM_ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={SLURM_CPUS}
#SBATCH --mem={SLURM_MEM}
#SBATCH --time={SLURM_TIME}
#SBATCH --output={logs_dir}/{job_name}_%A_%a.out
#SBATCH --error={logs_dir}/{job_name}_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={SLURM_MAIL_USER}

echo "Starting job array ${{SLURM_ARRAY_JOB_ID}}, task ${{SLURM_ARRAY_TASK_ID}} on $(hostname)"

if [ -f {MINIFORGE_PATH} ]; then
    source {MINIFORGE_PATH}
else
    echo "Miniforge not found!"
    exit 1
fi

export MPLBACKEND=Agg
export SLURM_SEED=${{SLURM_ARRAY_TASK_ID}}

conda activate {CONDA_ENV}

cd {current_dir}
python {cmd}

echo "Seed ${{SLURM_SEED}} completed."
'''

script_path = os.path.join(logs_dir, f"{job_name}.sh")
with open(script_path, "w") as f:
    f.write(slurm_script)

print(f"SLURM script: {script_path}")
print(f"Submitting array job (seeds 1-10)...")

result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(f"Error: {result.stderr}")