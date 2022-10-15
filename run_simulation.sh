#!/bin/bash
RUNS=9
USE_VAL_DATA=true

for DATASET in airport omdb
do
    for MAX_DIRTY_PROP in 0.5 0.3 0.1 0.05
    do
        echo "Dumping data.."
        python dump_processed_data.py --dataset $DATASET --max-clean-num 1000 --max-dirty-prop $MAX_DIRTY_PROP
        
        (cd learner && gunicorn -w "$(($(sysctl -n hw.physicalcpu)-1))" --bind 0.0.0.0:5000 --log-level info --timeout 240 api:app)&
        
        sleep 20
        cd ./trainer
        
        # for TRAINER_TYPE in full-oracle learning-oracle bayesian
        for TRAINER_TYPE in bayesian
        do
            for TRAINER_PRIOR_TYPE in data-estimate uniform-0.1 uniform-0.5 uniform-0.9 random
            do
                for LEARNER_PRIOR_TYPE in data-estimate uniform-0.1 uniform-0.5 uniform-0.9 random
                do
                    for SAMPLING_TYPE  in RANDOM ACTIVELR STOCHASTICBR STOCHASTICUS
                    do
                        echo "Running simulation for $DATASET $TRAINER_TYPE $SAMPLING_TYPE $RUNS $USE_VAL_DATA $TRAINER_PRIOR_TYPE $LEARNER_PRIOR_TYPE"
                        python simulate.py $DATASET $TRAINER_TYPE $SAMPLING_TYPE $RUNS $USE_VAL_DATA $TRAINER_PRIOR_TYPE $LEARNER_PRIOR_TYPE
                    done
                    
                done
            done
        done
        cd ..
        pkill -f gunicorn
        sleep 20
    done
done

