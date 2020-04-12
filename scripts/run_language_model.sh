#!/bin/bash
#SBATCH --time=180:00
#SBATCH --gres=gpu:k80:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G
#SBATCH --reservation=IFT6759

module load python/3.7
source /home/guest140/harman_venv/bin/activate

echo
echo ~~~~~~~~~~~~Running Language Model
echo

python run_language_modelling.py \
	--train_data_file ./data/merged_data.en \
	--output_dir ./models/roberta-v1 \
    --mlm \
    --model_type ./tokenizer_data \
    --config_name ./tokenizer_data \
    --tokenizer_name ./tokenizer_data \
    --do_train \
    --num_train_epochs 3 \
    --seed 42 \