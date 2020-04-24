## IFT6759: Low Resource Machine Translation

This project was created as part of the UdeM course IFT6759 (https://admission.umontreal.ca/cours-et-horaires/cours/IFT-6759/). The objective of this project is to predict French translations of English sentences in a low-resource setting. Refer to the report and presentation included in this repository for more details.

## To run the evaluation script:

Steps to run evaluation:

1. Go to `scripts` folder  
2. Edit `run_evaluator_script.sh`. Change the `--input-file-path` and `--target-file-path` as required  
3. Submit batch job: Run `sbatch run_evaluator_script.sh` from inside scripts folder


### K-Fold Strategy

* As we had small aligned dataset of 11k examples, we decided to use single-fold validation held out strategy.

### To setup a new local environment:

```console
module load python/3.7
virtualenv ../local_env
source ../local_env/bin/activate
pip install -r requirements_local.txt
```

