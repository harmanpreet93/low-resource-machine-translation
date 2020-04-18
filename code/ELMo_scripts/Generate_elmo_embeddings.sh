#!/bin/bash
#SBATCH --time=60:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10000M

source /home/guest142/env2/bin/activate


python Generate_en_elmo_embeddings.py


