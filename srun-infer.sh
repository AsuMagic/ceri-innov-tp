#!/bin/bash
#SBATCH -c 12
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH -p gpu
#SBATCH --job-name="RIP eris"
#SBATCH --gpus=1
#SBATCH --exclude=aura,talos
#SBATCH --constraint=GPURAM_Max_11GB

source ~/.bashrc
python3 infer.py --device=cuda --batch-size=8 $@
