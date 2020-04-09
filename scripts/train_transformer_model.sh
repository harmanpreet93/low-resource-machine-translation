#!/bin/bash
#SBATCH --time=120:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G

module load python/3.7
source /home/guest140/harman_venv/bin/activate

date
echo ~~~~~~~~~~~~Training Transformer model
echo

python ../code/train_transformer_model.py \
            --config ../code/user_config.json \
