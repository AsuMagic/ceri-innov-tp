#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH -p gpu
#SBATCH --job-name="RIP eris"
#SBATCH --gpus=1
#SBATCH --exclude=aura,talos
##SBATCH --nodelist=eris
#SBATCH --constraint=GPURAM_Max_11GB

source ~/.bashrc
python3 allocine.py train $@