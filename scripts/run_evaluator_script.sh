#!/bin/bash
#SBATCH --time=100:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
# --reservation=IFT6759_2020-04-10

module load python/3.7
source /home/guest140/harman_venv/bin/activate

echo
date
echo ~~~~~~~~Running Evaluator Script~~~~~~~~
echo

# dry run
python -u ../code/evaluator.py --input-file-path ../data/test_en.txt --target-file-path ../data/test_fr.txt


# if generating predictions from scratch, then run following command
#python -u ../evaluator.py --input-file-path inputs.txt --target-file-path targets.txt

# if predictions already created, run the following command. Note the --do-not-run-model flag
# python -u evaluator.py --input-file-path predictions.txt --target-file-path targets.txt --do-not-run-model



######################################################################################################
# Instructions from evaluation.md file from server:
######################################################################################################

# The quickest way to score your model output is to write the predictions to a
# file (say predictions.txt) and the targets to another file (say targets.txt) and run:

# python evaluator.py --input-file-path predictions.txt --target-file-path targets.txt --do-not-run-model

# Note the flag --do-not-run-model which means that the input file will be considered as predictions
# (instead of as inputs for your model).

# If you want to call your models directly inside the evaluator script instead, you should not use the
# flag --do-not-run-model and you should pass the input data (say inputs.txt) to the flag --input-file-path.
# For example:

# python evaluator.py --input-file-path inputs.txt --target-file-path targets.txt

# In this case, the script will call your model to generate prediction on the input file (inputs.txt),
# it will store the predictions in a temporary file, and it will use these predictions to compute the
# BLEU score. Note that for this to work, you will need to include (in the evaluator.py script) the code
# to run your model.



