#!/bin/bash
#SBATCH -c 10
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH -p gpu
#SBATCH --job-name="RIP eris"
#SBATCH --gpus=4
#SBATCH --exclude=aura,talos
##SBATCH --nodelist=eris
#SBATCH --constraint=GPURAM_Max_11GB

source ~/.bashrc
# export PATH=$PATH:/data/coros1/sdelangen/cuda/11.8/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/coros1/sdelangen/cuda/11.8/lib64
# export CUDA_HOME=/data/coros1/sdelangen/cuda/11.8/

python3 allocine.py hyperparam