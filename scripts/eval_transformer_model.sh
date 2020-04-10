#!/bin/bash
#SBATCH --time=300:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --reservation=IFT6759_2020-04-10

module load python/3.7
source /home/guest140/harman_venv/bin/activate

date
echo ~~~~~~~~~~~~Evaluating Transformer model
echo

python -u ../code/eval_transformer_model.py \
            --config ../code/user_config.json \
            --input_file_path ../data/test_en.txt \
            --pred_file_path ../log/pred_output.txt \
