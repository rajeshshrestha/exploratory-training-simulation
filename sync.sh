#!/bin/bash
cd /Users/rajeshshrestha/OSU/Codes/Research
rsync -uav --exclude={'*.venv','*.idea','*.vscode','*.ipynb_checkpoints','*store*','exploratory-training-simulation/learner/store','exploratory-training-simulation/trainer/trainer-store','*__pycache__*','*logs','*simulation-data','*/processed-exp-data'} ./exploratory-training-simulation shresthr@pelican03.eecs.oregonstate.edu:/nfs/stak/users/shresthr/OSU/Codes/Research/