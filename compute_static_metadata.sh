#!/bin/sh
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --partition=cpuonly
#SBATCH --job-name="sptokcalc"

srun python3 staticmetadata.py dataset/dev.bin.zst dataset/dev-metadata.bin.zst
srun python3 staticmetadata.py dataset/test.bin.zst dataset/test-metadata.bin.zst
srun python3 staticmetadata.py dataset/train.bin.zst dataset/train-metadata.bin.zst