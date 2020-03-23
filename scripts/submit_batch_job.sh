#!/bin/bash
#SBATCH --time=360:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10000M
#SBATCH --reservation=IFT6759_2020-02-26

date
echo ~~~~~~~~~~~~setting up environement
module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r ../requirements.txt

date
echo ~~~~~~~~~~~~starting training loop
python ../code/training_loop_launcher.py ../train_cfg.json ../val_cfg.json -u="../code/eval_user_cfg_cnn.json"

date
echo ~~~~~~~~~~~~finished
