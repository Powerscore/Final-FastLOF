#!/usr/bin/env python3
"""
SLURM Job Submission and Management Script for LOF Experiments
================================================================

This script:
1. Generates individual SLURM job scripts for each dataset
2. Submits jobs in batches to avoid overloading the queue
3. Monitors running jobs and submits new ones as slots become available
4. Provides real-time status updates

Supports both FastLOF and Original LOF experiments.

Usage:
    python slurm_submit.py [OPTIONS]

Options:
    --experiment-type TYPE  Experiment type: 'fastlof' (default) or 'original_lof'
    --generate-only        Generate job scripts without submitting
    --dry-run             Show what would be submitted without actually submitting
    --max-concurrent N     Set maximum concurrent jobs (overrides config)
    --dataset NAME         Submit only specific dataset(s)
    --resume               Resume from last submission state
"""

import os
import sys
import yaml
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional

class SlurmJobManager:
    """Manages SLURM job submission and monitoring for LOF experiments."""
    
    def __init__(self, experiment_type: str = "fastlof", config_path: str = "slurm_config.yaml"):
        self.experiment_type = experiment_type
        if self.experiment_type not in ["fastlof", "original_lof"]:
            raise ValueError(f"Invalid experiment_type: {experiment_type}. Must be 'fastlof' or 'original_lof'")
        
        self.config_path = config_path
        self.config = self._load_config()
        self.script_dir = Path(__file__).parent.absolute()
        self.log_dir = self.script_dir / "slurm_logs"
        self.jobs_dir = self.script_dir / "slurm_jobs"
        self.state_file = self.script_dir / f".slurm_state_{experiment_type}.yaml"
        
        # Job tracking
        self.submitted_jobs: Dict[str, str] = {}  # dataset_name -> job_id
        self.completed_jobs: Set[str] = set()
        self.failed_jobs: Set[str] = set()
        self.pending_datasets: List[str] = []
        
        # Ensure directories exist
        self.log_dir.mkdir(exist_ok=True)
        self.jobs_dir.mkdir(exist_ok=True)
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        config_path = Path(__file__).parent / self.config_path
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _save_state(self):
        """Save current submission state."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'submitted_jobs': self.submitted_jobs,
            'completed_jobs': list(self.completed_jobs),
            'failed_jobs': list(self.failed_jobs),
            'pending_datasets': self.pending_datasets
        }
        with open(self.state_file, 'w') as f:
            yaml.dump(state, f, default_flow_style=False)
    
    def _load_state(self) -> bool:
        """Load previous submission state if exists."""
        if not self.state_file.exists():
            return False
        
        with open(self.state_file, 'r') as f:
            state = yaml.safe_load(f)
        
        self.submitted_jobs = state.get('submitted_jobs', {})
        self.completed_jobs = set(state.get('completed_jobs', []))
        self.failed_jobs = set(state.get('failed_jobs', []))
        self.pending_datasets = state.get('pending_datasets', [])
        
        print(f"Loaded previous state from {state['timestamp']}")
        return True
    
    def generate_job_script(self, dataset_name: str) -> Path:
        """Generate SLURM job script for a specific dataset."""
        dataset_config = self.config['datasets'][dataset_name]
        global_config = self.config['global_settings']
        
        # Read template
        template_path = self.script_dir / "slurm_template.sh"
        with open(template_path, 'r') as f:
            template = f.read()
        
        # Prepare email lines
        email_lines = ""
        if global_config.get('email'):
            email_lines = f"#SBATCH --mail-user={global_config['email']}\n"
            email_lines += f"#SBATCH --mail-type={global_config.get('email_type', 'FAIL,END')}"
        
        # Determine script path based on experiment type
        if self.experiment_type == "original_lof":
            script_path = f"experiment_scripts/{dataset_name}/run_original_lof.py"
            experiment_display = "Original LOF"
        else:  # fastlof
            script_path = f"experiment_scripts/{dataset_name}/run_fastlof.py"
            experiment_display = "FastLOF"
        
        # Fill in template
        job_script = template.format(
            JOB_NAME=f"{self.experiment_type}_{dataset_name}",
            DATASET_NAME=dataset_name,
            EXPERIMENT_TYPE=self.experiment_type,
            EXPERIMENT_DISPLAY=experiment_display,
            TIME=dataset_config['time'],
            CPUS=dataset_config['cpus'],
            MEMORY=dataset_config['memory'],
            PARTITION=dataset_config['partition'],
            NUM_THREADS=global_config['num_threads'],
            ESTIMATED_TIME=dataset_config.get('estimated_time', 'unknown'),
            EMAIL_LINES=email_lines,
            WORK_DIR=global_config['work_dir'],
            PYTHON_CMD=global_config['python_cmd'],
            SCRIPT_PATH=script_path
        )
        
        # Save job script
        job_file = self.jobs_dir / f"{dataset_name}_{self.experiment_type}.sh"
        with open(job_file, 'w') as f:
            f.write(job_script)
        
        # Make executable
        os.chmod(job_file, 0o755)
        
        return job_file
    
    def generate_all_job_scripts(self):
        """Generate SLURM job scripts for all datasets."""
        print("=" * 80)
        print(f"Generating SLURM Job Scripts - {self.experiment_type.upper()}")
        print("=" * 80)
        print()
        
        datasets = self.config['datasets']
        for dataset_name in datasets:
            job_file = self.generate_job_script(dataset_name)
            print(f"[OK] Generated: {job_file.name}")
        
        print()
        print(f"Generated {len(datasets)} job scripts in {self.jobs_dir}")
        print()
    
    def submit_job(self, dataset_name: str, dry_run: bool = False) -> Optional[str]:
        """Submit a single job and return job ID."""
        job_file = self.jobs_dir / f"{dataset_name}_{self.experiment_type}.sh"
        
        if not job_file.exists():
            print(f"[ERROR] Job script not found: {job_file}")
            return None
        
        cmd = ['sbatch', str(job_file)]
        
        if dry_run:
            print(f"[DRY RUN] Would submit: {' '.join(cmd)}")
            return f"DRYRUN_{dataset_name}"
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            # Parse job ID from output like "Submitted batch job 12345"
            output = result.stdout.strip()
            job_id = output.split()[-1]
            
            print(f"[OK] Submitted {dataset_name}: Job ID {job_id}")
            return job_id
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to submit {dataset_name}: {e.stderr}")
            return None
    
    def get_running_jobs(self) -> Set[str]:
        """Get set of currently running job IDs."""
        try:
            result = subprocess.run(
                ['squeue', '-u', os.environ.get('USER', ''), '-h', '-o', '%i'],
                capture_output=True,
                text=True,
                check=True
            )
            job_ids = set(result.stdout.strip().split('\n'))
            return {jid for jid in job_ids if jid}  # Filter empty strings
        except subprocess.CalledProcessError:
            print("Warning: Could not query SLURM queue")
            return set()
    
    def get_job_status(self, job_id: str) -> str:
        """Get status of a specific job."""
        try:
            result = subprocess.run(
                ['squeue', '-j', job_id, '-h', '-o', '%T'],
                capture_output=True,
                text=True
            )
            status = result.stdout.strip()
            return status if status else 'COMPLETED'
        except subprocess.CalledProcessError:
            return 'UNKNOWN'
    
    def print_status(self):
        """Print current job status."""
        print()
        print("=" * 80)
        print(f"Job Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Count by status
        running = []
        for dataset, job_id in self.submitted_jobs.items():
            if dataset not in self.completed_jobs and dataset not in self.failed_jobs:
                running.append(dataset)
        
        print(f"Running:   {len(running)}")
        print(f"Completed: {len(self.completed_jobs)}")
        print(f"Failed:    {len(self.failed_jobs)}")
        print(f"Pending:   {len(self.pending_datasets)}")
        print()
        
        if running:
            print("Currently Running:")
            for dataset in running:
                job_id = self.submitted_jobs[dataset]
                est_time = self.config['datasets'][dataset].get('estimated_time', '?')
                print(f"  • {dataset:45} [{job_id}] ({est_time})")
            print()
        
        if self.pending_datasets[:5]:  # Show next 5
            print("Next in Queue:")
            for dataset in self.pending_datasets[:5]:
                est_time = self.config['datasets'][dataset].get('estimated_time', '?')
                print(f"  • {dataset:45} ({est_time})")
            if len(self.pending_datasets) > 5:
                print(f"  ... and {len(self.pending_datasets) - 5} more")
            print()
        
        print("=" * 80)
        print()
    
    def run_submission_loop(self, max_concurrent: int, dry_run: bool = False):
        """Main loop to manage job submissions."""
        print("=" * 80)
        print(f"Starting SLURM Job Submission Manager - {self.experiment_type.upper()}")
        print("=" * 80)
        print()
        print(f"Maximum concurrent jobs: {max_concurrent}")
        print(f"Check interval: {self.config['submission']['check_interval']} seconds")
        print()
        
        if dry_run:
            print("*** DRY RUN MODE - No jobs will actually be submitted ***")
            print()
        
        # Initial submissions
        while len(self.submitted_jobs) < max_concurrent and self.pending_datasets:
            dataset = self.pending_datasets.pop(0)
            job_id = self.submit_job(dataset, dry_run=dry_run)
            if job_id:
                self.submitted_jobs[dataset] = job_id
            else:
                self.failed_jobs.add(dataset)
            self._save_state()
        
        if dry_run:
            print("\nDry run complete. No actual jobs submitted.")
            return
        
        # Monitoring loop
        check_interval = self.config['submission']['check_interval']
        
        while self.pending_datasets or (len(self.submitted_jobs) > len(self.completed_jobs) + len(self.failed_jobs)):
            self.print_status()
            
            # Check status of submitted jobs
            running_job_ids = self.get_running_jobs()
            
            for dataset, job_id in list(self.submitted_jobs.items()):
                if dataset in self.completed_jobs or dataset in self.failed_jobs:
                    continue
                
                if job_id not in running_job_ids:
                    # Job finished - check if successful
                    # For simplicity, mark as completed if not in queue
                    # Could enhance this by checking exit codes in log files
                    self.completed_jobs.add(dataset)
                    print(f"[OK] {dataset} completed (Job {job_id})")
                    self._save_state()
            
            # Submit new jobs if slots available
            current_running = len(self.submitted_jobs) - len(self.completed_jobs) - len(self.failed_jobs)
            slots_available = max_concurrent - current_running
            
            while slots_available > 0 and self.pending_datasets:
                dataset = self.pending_datasets.pop(0)
                job_id = self.submit_job(dataset)
                if job_id:
                    self.submitted_jobs[dataset] = job_id
                    slots_available -= 1
                else:
                    self.failed_jobs.add(dataset)
                self._save_state()
            
            if self.pending_datasets or current_running > 0:
                print(f"Checking again in {check_interval} seconds...")
                time.sleep(check_interval)
        
        # Final status
        print()
        print("=" * 80)
        print("All Jobs Complete!")
        print("=" * 80)
        print()
        print(f"Completed: {len(self.completed_jobs)}/{len(self.config['datasets'])}")
        if self.failed_jobs:
            print(f"Failed: {len(self.failed_jobs)}")
            print("Failed datasets:")
            for dataset in self.failed_jobs:
                print(f"  • {dataset}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="SLURM job submission manager for LOF experiments (FastLOF and Original LOF)"
    )
    parser.add_argument(
        '--experiment-type',
        type=str,
        default='fastlof',
        choices=['fastlof', 'original_lof'],
        help='Experiment type: fastlof (default) or original_lof'
    )
    parser.add_argument(
        '--generate-only',
        action='store_true',
        help='Only generate job scripts, do not submit'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be submitted without actually submitting'
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        help='Maximum concurrent jobs (overrides config)'
    )
    parser.add_argument(
        '--dataset',
        action='append',
        help='Submit only specific dataset(s), can be used multiple times'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last submission state'
    )
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = SlurmJobManager(experiment_type=args.experiment_type)
    
    # Generate job scripts
    manager.generate_all_job_scripts()
    
    if args.generate_only:
        print("Job scripts generated. Use --submit to submit them.")
        return
    
    # Determine datasets to submit
    if args.dataset:
        datasets_to_submit = args.dataset
        # Validate dataset names
        for ds in datasets_to_submit:
            if ds not in manager.config['datasets']:
                print(f"Error: Unknown dataset '{ds}'")
                print(f"Available datasets: {', '.join(manager.config['datasets'].keys())}")
                sys.exit(1)
    else:
        # Get all datasets sorted by priority
        datasets = manager.config['datasets']
        datasets_to_submit = sorted(
            datasets.keys(),
            key=lambda x: datasets[x].get('priority', 999)
        )
    
    # Handle resume
    if args.resume:
        if manager._load_state():
            # Remove completed/failed from list
            datasets_to_submit = [
                ds for ds in datasets_to_submit
                if ds not in manager.completed_jobs and ds not in manager.failed_jobs
            ]
        else:
            print("No previous state found, starting fresh.")
    
    manager.pending_datasets = datasets_to_submit
    
    # Get max concurrent from args or config
    max_concurrent = args.max_concurrent or manager.config['submission']['max_concurrent_jobs']
    
    # Run submission loop
    try:
        manager.run_submission_loop(max_concurrent, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. State saved.")
        print("Use --resume to continue from where you left off.")
        manager._save_state()
        sys.exit(0)


if __name__ == "__main__":
    main()
