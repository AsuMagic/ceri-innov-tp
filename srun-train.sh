#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH -p gpu
#SBATCH --job-name="trainac"
#SBATCH --gpus=1
#SBATCH --exclude=aura,talos
#SBATCH --nodelist=eris
##SBATCH --constraint=GPURAM_Max_11GB

source ~/.bashrc
# export PATH=$PATH:/data/coros1/sdelangen/cuda/11.8/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/coros1/sdelangen/cuda/11.8/lib64
# export CUDA_HOME=/data/coros1/sdelangen/cuda/11.8/

python3 allocine.py model.camembert-base.ckpt \
    --camembert-model="camembert-base" \
    --camembert-removed-layers=3 \
    --train-path="dataset/camembert-base/train-metadata.bin.zst" \
    --dev-path="dataset/camembert-base/dev-metadata.bin.zst" \
    $@

# python3 allocine.py model.camembert-large.ckpt \
#     --camembert-model="camembert/camembert-large" \
#     --camembert-removed-layers=3 \
#     --train-path="dataset/camembert-large/train-metadata.bin.zst" \
#     --dev-path="dataset/camembert-large/dev-metadata.bin.zst" \
#     $@
