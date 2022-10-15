#!/bin/bash
RUNS=9
USE_VAL_DATA=true

for MAX_DIRTY_PROP in 0.05 0.1 0.2 0.3 0.5
do
    echo "Dumping data.."
    python dump_processed_data.py --max-dirty-prop $MAX_DIRTY_PROP
    
    (cd learner && gunicorn -w "$(sysctl -n hw.physicalcpu)" --bind 0.0.0.0:5000 --log-level info --timeout 240 api:app)&
    
    sleep 10
    cd ./trainer
    
    # for TRAINER_TYPE in full-oracle learning-oracle bayesian
    for TRAINER_TYPE in bayesian
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
    cd ..
    pkill -f gunicorn
    sleep 15
done

