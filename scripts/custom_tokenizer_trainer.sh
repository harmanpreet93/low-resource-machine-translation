#!/bin/bash
#SBATCH --time=30:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G

module load python/3.7
source /home/guest140/harman_venv/bin/activate

date
echo ~~~~~~~~~~~~Tokenizing English data
echo

# change training_folder and path_to_save_tokenizer according to language tokenizer to train
# training_folder contains file only related to particular language.
# For example, if training an english tokenizer, training_folder will contain - unaligned.en and train_en.lang1

python ../code/custom_tokenizer_trainer.py \
            --vocab_size 30000 \
            --lowercase True \
            --min_frequency 2 \
            --training_folder ../../tokenizer_train_folder_en \
            --path_to_save_tokenizer ../tokenizer_data_en_30k \


echo
echo ~~~~~~~~~~~~Tokenizing French data
echo

# Note: Keep lowercase=False for french

python ../code/custom_tokenizer_trainer.py \
            --vocab_size 30000 \
            --lowercase False \
            --min_frequency 2 \
            --training_folder ../../tokenizer_train_folder_fr \
            --path_to_save_tokenizer ../tokenizer_data_fr_30k \