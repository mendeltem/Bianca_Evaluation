#!/bin/bash
#SBATCH --job-name=run_removal

#SBATCH --account=sc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/DATASET_SC/PROJECT_NULL_07_04_2025/1_Project_all_code_all_data/log/run_removal_%A_%a.out
#SBATCH --error=/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/DATASET_SC/PROJECT_NULL_07_04_2025/1_Project_all_code_all_data/log/run_removal_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=uchralt.temuulen@charite.de

echo "Starting job array ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID} on $(hostname)"

if [ -f /opt/miniforge/etc/profile.d/conda.sh ]; then
    source /opt/miniforge/etc/profile.d/conda.sh
else
    echo "Miniforge not found!"
    exit 1
fi

export MPLBACKEND=Agg

conda activate /home/temuuleu/SC_Stroke_MRI/Software/miniconda3/envs/bianca_env

python /sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/DATASET_SC/PROJECT_NULL_07_04_2025/1_Project_all_code_all_data/7_run_removal.py

echo "Job completed."
