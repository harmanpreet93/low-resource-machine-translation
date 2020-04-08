#!/bin/bash
#SBATCH --time=10:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --reservation=IFT6759

module load python/3.7
source /home/guest140/harman_venv/bin/activate
# source /home/guest164/ift6759-env/bin/activate

date
echo ~~~~~~~~~~~~Tokenizing data
echo

python custom_tokenizer_trainer.py \
	--vocab_size 60000 \
	--lowercase True \
	--min_frequency 2 \
	--training_folder ./tokenizer_train_folder_en \
	--path_to_save_tokenizer ./tokenizer_data \