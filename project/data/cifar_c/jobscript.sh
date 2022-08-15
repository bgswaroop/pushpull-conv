#!/bin/bash
#SBATCH --job-name=cifar
#SBATCH --time=24:00:00
#SBATCH --mem=64g
#SBATCH --cpus-per-task=5
#SBATCH --array=0-7
#SBATCH --mail-user=g.s.bennabhaktula@rug.nl
#SBATCH --mail-type=FAIL

# SLURM Notation used above
# %x - Name of the Job
# %A - JOB ID
# %a - TASK ID

source /data/p288722/python_venv/deep_hashing/bin/activate

python /home/p288722/git_code/deep_hashing/project/data/cifar_c/make_cifar_c.py --index ${SLURM_ARRAY_TASK_ID}