#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --time=0:30:00
#SBATCH --mem=4g
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpushort
#SBATCH --cpus-per-task=2

# SLURM Notation used above
# %x - Name of the Job
# %A - JOB ID
# %a - TASK ID

SECONDS=0;
tar -xf /scratch/p288722/data/imagenet-c.tar -C $TMPDIR --warning=no-unknown-keyword
echo echo Time taken to extract imagenet-c $SECONDS sec

SECONDS=0;
tar -xf /scratch/p288722/data/imagenet.tar -C $TMPDIR --warning=no-unknown-keyword
echo echo Time taken to extract imagenet $SECONDS sec

cd $TMPDIR || exit
pwd
ls -alh $TMPDIR
