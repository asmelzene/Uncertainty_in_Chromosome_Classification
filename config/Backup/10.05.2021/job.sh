#!/bin/sh
#SBATCH -n 16
#SBATCH -t 02:30:00  # 2 hours and 30 minutes
#SBATCH -J test_job_melih # sensible name for the job

# load the modules
#conda load apps staskfarm

# we need Python 2.7
#module load gcc python/2.7.18

# execute the commands via the slurm task farm wrapper
#staskfarm commands.txt
conda info
