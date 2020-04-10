#!/bin/bash
#SBATCH --time=60:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G

module load python/3.7
source /home/guest140/harman_venv/bin/activate

date
echo ~~~~~~~~~~~~Running Evaluator script
echo

python -u ../code/evaluator.py \
            --input-file-path ../log/predicted_fr_1.txt \
            --target-file-path ../log/true_fr_1.txt \
            --do-not-run-model \