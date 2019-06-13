#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=gen-emi-embedding
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6GB
#SBATCH --time=2:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output="gen-emi-embedding-%A-%a.out"
#SBATCH --err="gen-emi-embedding-%A-%a.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

source activate tfgpu
module purge
module load cudnn/9.0v7.3.0.29

SRCDIR=$HOME/EdgeML
data_dir=$1
DATASET_DIR=/scratch/dr2915/Bumblebee/${data_dir}
NUM_OUTPUT=$2
NUM_HIDDEN=$3
K=$4
GATE_NL='sigmoid'
UPDATE_NL='tanh'
BATCH_SIZE=$5
ORIGINAL_NUM_TIMESTEPS=$(echo ${data_dir} | cut -d'_' -f4)
NUM_EPOCHS=10
NUM_ITER=5
NUM_ROUNDS=10

python $SRCDIR/tf/examples/EMI-RNN/step3_get_emi_fastgrnn_embeddings_tvt.py\
    -Dat $DATASET_DIR\
    -O $NUM_OUTPUT\
    -ots $ORIGINAL_NUM_TIMESTEPS\
    -k $K\
    -H $NUM_HIDDEN\
    -gN $GATE_NL\
    -uN $UPDATE_NL\
    -bs $BATCH_SIZE\
    -ep $NUM_EPOCHS\
    -it $NUM_ITER\
    -rnd $NUM_ROUNDS
