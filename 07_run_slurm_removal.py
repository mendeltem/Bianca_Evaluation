import os
import subprocess
from dotenv import load_dotenv

load_dotenv()
# Load from .env
#pip install dotenv

SLURM_ACCOUNT    = os.getenv("SLURM_ACCOUNT", "sc-users")
SLURM_MAIL_USER  = os.getenv("SLURM_MAIL_USER")
PARTITION_TYPE    = os.getenv("SLURM_PARTITION_TYPE", "cpu")
SLURM_CPUS       = os.getenv("SLURM_CPUS", "8")
SLURM_MEM        = os.getenv("SLURM_MEM", "128G")
SLURM_TIME       = os.getenv("SLURM_TIME", "48:00:00")
CONDA_ENV        = os.getenv("CONDA_ENV", "/home/temuuleu/SC_Stroke_MRI/Software/miniconda3/envs/bianca_env")
MINIFORGE_PATH   = os.getenv("MINIFORGE_PATH", "/opt/miniforge/etc/profile.d/conda.sh")

current_dir = os.getcwd()
cmd = os.path.join(current_dir, "7_run_removal.py")
logs_dir = os.path.join(os.path.dirname(cmd), "log")
os.makedirs(logs_dir, exist_ok=True)

job_name = f"run_removal"

if PARTITION_TYPE == "gpu":
    partition = """#SBATCH --partition=gpu
#SBATCH --gres=shard:1"""
else:
    partition = ""

slurm_script = f'''#!/bin/bash
#SBATCH --job-name={job_name}
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

conda activate {CONDA_ENV}

python {cmd}

echo "Job completed."
'''

script_file = "7_run_removal.sh"
with open(script_file, "w") as f:
    f.write(slurm_script)

os.chmod(script_file, 0o755)

print(f"\n👉 Submitting job...")
result = subprocess.run(["sbatch", script_file], capture_output=True, text=True)
if result.returncode == 0:
    print(f"✅ Success! {result.stdout.strip()}")
else:
    print(f"❌ Error: {result.stderr}")