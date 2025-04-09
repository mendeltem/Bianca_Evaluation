#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description="Submit a SLURM job array to run a Python script with multiple .env configs.")
    parser.add_argument("--script", type=str, default="smriprep-preprocessed.py",
                        help="Path to the Python script to run.")
    parser.add_argument("--conda_env", type=str, default="/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/Software/miniconda3/envs/mb",
                        help="Path to the conda environment to activate.")
    parser.add_argument("--bash_script", type=str, default="run_smriprep_array.sh",
                        help="Filename for the generated SLURM bash script.")
    parser.add_argument("--env_dir", type=str, default="envs",
                        help="Directory containing the .env files.")

    args = parser.parse_args()

    # Gather all .env files
    env_files = sorted(glob.glob(os.path.join(args.env_dir, "*.env")))
    num_envs = len(env_files)

    if num_envs == 0:
        print(f"Keine .env-Dateien gefunden in {args.env_dir}")
        return

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Create SLURM bash script
    bash_script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --job-name=multi-env-run
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-{num_envs - 1}

echo "Starte SLURM-Job mit Array ID $SLURM_ARRAY_TASK_ID"

env_files=({ ' '.join(env_files) })
env_file=${{env_files[$SLURM_ARRAY_TASK_ID]}}

if [ -f /opt/miniforge/etc/profile.d/conda.sh ]; then
    source /opt/miniforge/etc/profile.d/conda.sh
else
    echo "Miniforge nicht gefunden!"
    exit 1
fi

conda activate {args.conda_env}
export PATH={args.conda_env}/bin:$PATH

echo "Aktivierte Umgebung:"
conda info --envs
echo "Python-Pfad:"
which python
echo "Python-Version:"
python --version

echo "Starte Python-Code mit env-file $env_file..."
python {os.path.abspath(args.script)} $env_file
echo "Job abgeschlossen."
"""

    # Write bash script
    with open(args.bash_script, "w") as f:
        f.write(bash_script)

    # Make script executable
    os.chmod(args.bash_script, 0o755)

    # Submit with sbatch
    print(f"Starte SLURM-Array-Job für {num_envs} .env-Dateien...")
    result = subprocess.run(["sbatch", args.bash_script], capture_output=True, text=True)

    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)

if __name__ == "__main__":
    main()
