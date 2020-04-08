#!/bin/bash
#SBATCH --time=60:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --reservation=IFT6759

module load python/3.7
source /home/guest140/harman_venv/bin/activate

date
echo ~~~~~~~~~~~~Tokenizing data
echo

# change training_folder and path_to_save_tokenizer according to language tokenizer to train
# training_folder contains file only related to particular language.
# For example, if training an english tokenizer, training_folder will contain - unaligned.en and train.lang1

# Note: Keep lowercase=False for french

python ../code/custom_tokenizer_trainer.py \
            --vocab_size 40000 \
            --lowercase True \
            --min_frequency 2 \
            --training_folder ../../pretrain_language_model/tokenizer_train_folder_en \
            --path_to_save_tokenizer ../tokenizer_data_en \