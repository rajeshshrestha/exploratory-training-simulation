#!/bin/sh
cd /Users/rajeshshrestha/OSU/Codes/Research
rsync -uav --exclude '*.venv' --exclude '*.idea' --exclude '*.vscode' --exclude '*.ipynb_checkpoints' --exclude 'exploratory-training-simulation/learner/store*' --exclude '*__pycache__*' ./exploratory-training-simulation shresthr@pelican03.eecs.oregonstate.edu:/nfs/stak/users/shresthr/OSU/Codes/Research/