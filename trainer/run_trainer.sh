#!/bin/bash
RUNS=9
USE_VAL_DATA=true
TRAINER_PRIOR_TYPE=uniform-0.1
LEARNER_PRIOR_TYPE=data-estimate

for MAX_DIRTY_PROP in 0.05 0.1 0.2 0.3 0.5
do
    echo "Dumping data.."
    python dump_processed_data.py --max-dirty-prop $MAX_DIRTY_PROP
    for TRAINER_TYPE in full-oracle learning-oracle bayesian
    do
        for TRAINER_PRIOR_TYPE in uniform-0.1 uniform-0.5 uniform-0.9 random data-estimate
        do
            for LEARNER_PRIOR_TYPE in uniform-0.1 uniform-0.5 uniform-0.9 random data-estimate
            do
                for SAMPLING_TYPE  in RANDOM ACTIVELR STOCHASTICBR STOCHASTICUS
                do
                    echo "Running simulation for omdb $TRAINER_TYPE $SAMPLING_TYPE $RUNS $USE_VAL_DATA $TRAINER_PRIOR_TYPE $LEARNER_PRIOR_TYPE"
                    python simulate.py omdb $TRAINER_TYPE $SAMPLING_TYPE $RUNS $USE_VAL_DATA $TRAINER_PRIOR_TYPE $LEARNER_PRIOR_TYPE
                done
                
            done
        done
    done
done

