"""
================================================================================
SLURM Job Submission Script for BIANCA Pipeline
================================================================================

Paper Reference: "Robustness and Error Susceptibility of BIANCA for White 
Matter Hyperintensity Segmentation"

This script submits the preprocessing and evaluation pipelines to a SLURM 
cluster for batch processing.

Cluster Requirements:
- SLURM workload manager
- FSL installed and available
- HD-BET installed (for brain extraction)
- Python environment with required packages

Usage:
    python slurm_submit_pipeline.py [script_name] [options]
    
    Options:
    --partition    : SLURM partition (default: gpu)
    --time         : Wall time limit (default: 12:00:00)
    --mem          : Memory allocation (default: 32G)
    --cpus         : CPUs per task (default: 8)

Example:
    python slurm_submit_pipeline.py 01_preprocessing_pipeline.py --partition=compute

Author: Uchralt Temuulen
================================================================================
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Default SLURM parameters
DEFAULT_CONFIG = {
    'partition': 'gpu',
    'gres': 'shard:1',
    'account': 'sc-users',
    'nodes': 1,
    'ntasks': 1,
    'cpus_per_task': 8,
    'memory': '32G',
    'time': '12:00:00',
    'mail_type': 'END,FAIL',
    'mail_user': ''  # Set your email here
}

# Conda environment path
CONDA_ENV = os.environ.get(
    'CONDA_ENV_PATH',
    '/path/to/conda/envs/bianca_env'
)

# Project directories
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.getcwd())
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')


# ==============================================================================
# SLURM SCRIPT TEMPLATE
# ==============================================================================

SLURM_TEMPLATE = '''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --account={account}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={memory}
#SBATCH --time={time}
#SBATCH --output={logs_dir}/{job_name}_%A_%a.out
#SBATCH --error={logs_dir}/{job_name}_%A_%a.err
{mail_config}

echo "=========================================="
echo "BIANCA Pipeline Job Submission"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# Load conda environment
if [ -f /opt/miniforge/etc/profile.d/conda.sh ]; then
    source /opt/miniforge/etc/profile.d/conda.sh
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "Error: Conda initialization script not found!"
    exit 1
fi

# Activate environment
conda activate {conda_env}

# Verify environment
echo "Python: $(which python)"
echo "FSL: $FSLDIR"

# Set environment variables
export PROJECT_ROOT="{project_root}"
export DATA_DIR="{data_dir}"
export OUTPUT_DIR="{output_dir}"

# Run the pipeline script
echo ""
echo "Running: {script_path}"
echo ""

python {script_path} {script_args}

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Job completed successfully"
    echo "End time: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Job failed with error"
    echo "End time: $(date)"
    echo "=========================================="
    exit 1
fi
'''


# ==============================================================================
# FUNCTIONS
# ==============================================================================

def create_slurm_script(
    script_path: str,
    config: dict,
    script_args: str = ""
) -> str:
    """
    Generate SLURM batch script content.
    
    Parameters
    ----------
    script_path : str
        Path to Python script to execute
    config : dict
        SLURM configuration parameters
    script_args : str
        Additional arguments for the script
        
    Returns
    -------
    str
        SLURM script content
    """
    job_name = Path(script_path).stem
    
    # Mail configuration
    mail_config = ""
    if config.get('mail_user'):
        mail_config = f"#SBATCH --mail-type={config['mail_type']}\n#SBATCH --mail-user={config['mail_user']}"
    
    # Format template
    slurm_content = SLURM_TEMPLATE.format(
        job_name=job_name,
        partition=config.get('partition', DEFAULT_CONFIG['partition']),
        gres=config.get('gres', DEFAULT_CONFIG['gres']),
        account=config.get('account', DEFAULT_CONFIG['account']),
        nodes=config.get('nodes', DEFAULT_CONFIG['nodes']),
        ntasks=config.get('ntasks', DEFAULT_CONFIG['ntasks']),
        cpus_per_task=config.get('cpus_per_task', DEFAULT_CONFIG['cpus_per_task']),
        memory=config.get('memory', DEFAULT_CONFIG['memory']),
        time=config.get('time', DEFAULT_CONFIG['time']),
        logs_dir=config.get('logs_dir', LOGS_DIR),
        mail_config=mail_config,
        conda_env=config.get('conda_env', CONDA_ENV),
        project_root=config.get('project_root', PROJECT_ROOT),
        data_dir=config.get('data_dir', ''),
        output_dir=config.get('output_dir', ''),
        script_path=script_path,
        script_args=script_args
    )
    
    return slurm_content


def submit_job(
    script_path: str,
    config: dict = None,
    script_args: str = "",
    dry_run: bool = False
) -> bool:
    """
    Submit a job to SLURM.
    
    Parameters
    ----------
    script_path : str
        Path to Python script to execute
    config : dict, optional
        SLURM configuration parameters
    script_args : str
        Additional arguments for the script
    dry_run : bool
        If True, print script without submitting
        
    Returns
    -------
    bool
        True if submission successful
    """
    config = config or DEFAULT_CONFIG.copy()
    
    # Create logs directory
    logs_dir = config.get('logs_dir', LOGS_DIR)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate SLURM script
    slurm_content = create_slurm_script(script_path, config, script_args)
    
    # Write temporary script file
    job_name = Path(script_path).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    slurm_script_path = os.path.join(logs_dir, f"{job_name}_{timestamp}.sh")
    
    with open(slurm_script_path, 'w') as f:
        f.write(slurm_content)
    
    os.chmod(slurm_script_path, 0o755)
    
    print(f"\n{'='*60}")
    print(f"SLURM Job Submission")
    print(f"{'='*60}")
    print(f"Script: {script_path}")
    print(f"SLURM file: {slurm_script_path}")
    print(f"Partition: {config.get('partition')}")
    print(f"Memory: {config.get('memory')}")
    print(f"Time limit: {config.get('time')}")
    print(f"{'='*60}")
    
    if dry_run:
        print("\n[DRY RUN] SLURM script content:")
        print("-" * 40)
        print(slurm_content)
        print("-" * 40)
        return True
    
    # Submit job
    try:
        result = subprocess.run(
            ["sbatch", slurm_script_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            job_id = result.stdout.strip()
            print(f"\n✓ Job submitted successfully!")
            print(f"  {job_id}")
            return True
        else:
            print(f"\n✗ Job submission failed!")
            print(f"  Error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("\n✗ Error: sbatch command not found.")
        print("  Make sure you are running on a SLURM cluster.")
        return False


# ==============================================================================
# PIPELINE SUBMISSION FUNCTIONS
# ==============================================================================

def submit_preprocessing(config: dict = None, dry_run: bool = False):
    """Submit preprocessing pipeline job."""
    script_path = os.path.join(PROJECT_ROOT, '01_preprocessing_pipeline.py')
    return submit_job(script_path, config, dry_run=dry_run)


def submit_evaluation(config: dict = None, dry_run: bool = False):
    """Submit evaluation pipeline job."""
    script_path = os.path.join(PROJECT_ROOT, '02_bianca_evaluation_pipeline.py')
    return submit_job(script_path, config, dry_run=dry_run)


def submit_loo_evaluation(config: dict = None, dry_run: bool = False):
    """Submit LOO cross-validation evaluation job."""
    script_path = os.path.join(PROJECT_ROOT, 'loo_evaluation_pipeline.py')
    
    # LOO typically needs more time
    if config is None:
        config = DEFAULT_CONFIG.copy()
    config['time'] = '24:00:00'
    config['memory'] = '64G'
    
    return submit_job(script_path, config, dry_run=dry_run)


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Submit BIANCA pipeline jobs to SLURM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python slurm_submit_pipeline.py 01_preprocessing_pipeline.py
  python slurm_submit_pipeline.py 02_bianca_evaluation_pipeline.py --partition=compute
  python slurm_submit_pipeline.py --all --dry-run
        """
    )
    
    parser.add_argument(
        'script',
        nargs='?',
        help='Python script to submit'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Submit all pipeline stages'
    )
    
    parser.add_argument(
        '--partition',
        default=DEFAULT_CONFIG['partition'],
        help=f"SLURM partition (default: {DEFAULT_CONFIG['partition']})"
    )
    
    parser.add_argument(
        '--time',
        default=DEFAULT_CONFIG['time'],
        help=f"Wall time limit (default: {DEFAULT_CONFIG['time']})"
    )
    
    parser.add_argument(
        '--mem',
        default=DEFAULT_CONFIG['memory'],
        help=f"Memory allocation (default: {DEFAULT_CONFIG['memory']})"
    )
    
    parser.add_argument(
        '--cpus',
        type=int,
        default=DEFAULT_CONFIG['cpus_per_task'],
        help=f"CPUs per task (default: {DEFAULT_CONFIG['cpus_per_task']})"
    )
    
    parser.add_argument(
        '--email',
        default='',
        help='Email for job notifications'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print SLURM script without submitting'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Build configuration from arguments
    config = DEFAULT_CONFIG.copy()
    config['partition'] = args.partition
    config['time'] = args.time
    config['memory'] = args.mem
    config['cpus_per_task'] = args.cpus
    config['mail_user'] = args.email
    
    if args.all:
        # Submit all pipeline stages
        print("\n" + "=" * 60)
        print("Submitting all pipeline stages...")
        print("=" * 60)
        
        submit_preprocessing(config, dry_run=args.dry_run)
        submit_evaluation(config, dry_run=args.dry_run)
        
    elif args.script:
        # Submit specific script
        if not os.path.exists(args.script):
            print(f"Error: Script not found: {args.script}")
            sys.exit(1)
        
        submit_job(args.script, config, dry_run=args.dry_run)
        
    else:
        print("Error: Please specify a script or use --all")
        print("Use --help for usage information")
        sys.exit(1)


if __name__ == "__main__":
    main()