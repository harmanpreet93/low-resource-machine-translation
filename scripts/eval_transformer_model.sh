#!/bin/bash
#SBATCH --time=300:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
# --reservation=IFT6759_2020-04-10

module load python/3.7
source /home/guest140/harman_venv/bin/activate

date
echo ~~~~~~~~~~~~Evaluating Transformer model
echo

export CUDA_VISIBLE_DEVICES=1
echo "Using GPU 1 to evaluate"
python -u ../code/generate_model_predictions.py \
            --config ../code/user_config.json \
            --input_file_path ../data/test_en.txt \
            --target_file_path ../data/test_fr.txt \
            --pred_file_path ../data/en_fr_test_en.txt \
            2>&1 | tee -a ../log/log.en_fr_test_en.log \
            