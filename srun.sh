#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH -p gpu
#SBATCH --job-name="trainac"
#SBATCH --gpus=1
#SBATCH --exclude=aura,talos
#SBATCH --constraint=GPURAM_Max_11GB

source ~/.bashrc
python3 allocine.py model.ckpt $@
