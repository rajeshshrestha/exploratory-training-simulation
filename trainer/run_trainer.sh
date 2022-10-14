#!/bin/bash
RUNS=10
python simulate.py omdb full-oracle RANDOM $RUNS;
python simulate.py omdb learning-oracle RANDOM $RUNS;
python simulate.py omdb uninformed-bayesian RANDOM $RUNS;
python simulate.py omdb full-oracle ACTIVELR $RUNS;
python simulate.py omdb learning-oracle ACTIVELR $RUNS;
python simulate.py omdb uninformed-bayesian ACTIVELR $RUNS;
python simulate.py omdb full-oracle STOCHASTICBR $RUNS;
python simulate.py omdb learning-oracle STOCHASTICBR $RUNS;
python simulate.py omdb uninformed-bayesian STOCHASTICBR $RUNS;
python simulate.py omdb full-oracle STOCHASTICUS $RUNS;
python simulate.py omdb learning-oracle STOCHASTICUS $RUNS;
python simulate.py omdb uninformed-bayesian STOCHASTICUS $RUNS;
