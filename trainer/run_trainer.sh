#!/bin/bash
RUNS=20
python simulate.py omdb full-oracle RANDOM $RUNS;
python simulate.py omdb learning-oracle RANDOM $RUNS;
python simulate.py omdb uninformed-bayesian RANDOM $RUNS;
python simulate.py omdb full-oracle ACTIVELR $RUNS;
python simulate.py omdb learning-oracle ACTIVELR $RUNS;
python simulate.py omdb uninformed-bayesian ACTIVELR $RUNS;
