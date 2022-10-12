#!/bin/bash
cd /Users/rajeshshrestha/OSU/Codes/Research
rsync -uav --exclude={'*.venv','*.idea','*.vscode','*.ipynb_checkpoints','exploratory-training-simulation/learner/store*','*__pycache__*','*logs','*simulation-data'} ./exploratory-training-simulation shresthr@pelican03.eecs.oregonstate.edu:/nfs/stak/users/shresthr/OSU/Codes/Research/