#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5GB
#SBATCH --mail-type=FAIL

source activate l3embedding-tf-12-gpu
#source activate tfgpu
module purge
module load cudnn/9.0v7.3.0.29
bash test_msc_trained_emi_embeddings_H=64_winlen=256.sh
