#!/bin/bash
#SBATCH --time=240:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

module load python/3.7
source /home/guest141/test_env/bin/activate

date
echo ~~~~~~~~~~~~Training GRU model
echo

python -u /home/guest141/GRU.py --config /home/guest141/data_project2/GRU_config.json --embedding_model Word2Vec\

date
echo ~~~~~~~~~~~~finished