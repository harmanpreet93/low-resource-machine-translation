#!/bin/bash
#SBATCH --time=720:00
#SBATCH --gres=gpu:k80:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=10000M

source /home/guest142/env2/bin/activate
export CUDA_VISIBLE_DEVICES=0,1

python /home/guest142/Project2/bilm-tf/bin/train_elmo.py --train_prefix='/project/cq-training-1/project2/teams/team08/ELMo/swb_fr/train/*' --vocab_file '/project/cq-training-1/project2/teams/team08/ELMo/swb_fr/vocab.txt' --save_dir '/project/cq-training-1/project2/teams/team08/ELMo/swb_fr/checkpoint'

