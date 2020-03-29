## Low Resource Machine Translation - IFT6759

This project was created as part of the UdeM course IFT6759 (https://admission.umontreal.ca/cours-et-horaires/cours/IFT-6759/). The objective of this project is to predict French translations of English sentences using a small dataset. Refer to the report and presentation included in this reporistory for more details.

### Team 08
* Alexander Peplowski
* Harmanpreet Singh
* Marc-Antoine Provost
* Mohammed Loukili


### To run the evaluation script:

```console
1. TBD
2. TBD
```

### K-Fold Strategy

* TBD

### Coding Standards

* Lint your code as per PEP8 before submitting a pull request
* Pull requests are required for merging to master for major changes
* Use your own branch for major work, don't use master
* No large files allowed in git
* Mark task in progress on Kanban before starting work

### To setup a new local environment:

```console
module load python/3.7
virtualenv ../local_env
source ../local_env/bin/activate
pip install -r requirements_local.txt
```

### To setup a new server node environment:

```console
module load python/3.7
virtualenv ../server_env --no-download
source ../server_env/bin/activate
pip install --no-index -r requirements.txt
```
OR, if no requirement.txt file is available:
```console
pip install --no-index tensorflow-gpu==2 pandas numpy tqdm
```

### To evaluate results from server locally using tensorboard:

Run the commands to synchronize data from the server and to launch tensorboard:
```console
./scripts/rsync_data.sh
./scripts/run_tensorboard.sh
```
Use a web browser to visit: http://localhost:6006/

