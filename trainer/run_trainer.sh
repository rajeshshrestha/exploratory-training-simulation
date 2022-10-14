#!/bin/bash
RUNS=9
USE_VAL_DATA=true
python simulate.py omdb full-oracle RANDOM $RUNS $USE_VAL_DATA;
python simulate.py omdb learning-oracle RANDOM $RUNS $USE_VAL_DATA;
python simulate.py omdb uninformed-bayesian RANDOM $RUNS $USE_VAL_DATA;
python simulate.py omdb full-oracle ACTIVELR $RUNS $USE_VAL_DATA;
python simulate.py omdb learning-oracle ACTIVELR $RUNS $USE_VAL_DATA;
python simulate.py omdb uninformed-bayesian ACTIVELR $RUNS $USE_VAL_DATA;
python simulate.py omdb full-oracle STOCHASTICBR $RUNS $USE_VAL_DATA;
python simulate.py omdb learning-oracle STOCHASTICBR $RUNS $USE_VAL_DATA;
python simulate.py omdb uninformed-bayesian STOCHASTICBR $RUNS $USE_VAL_DATA;
python simulate.py omdb full-oracle STOCHASTICUS $RUNS $USE_VAL_DATA;
python simulate.py omdb learning-oracle STOCHASTICUS $RUNS $USE_VAL_DATA;
python simulate.py omdb uninformed-bayesian STOCHASTICUS $RUNS $USE_VAL_DATA;
