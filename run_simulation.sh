#!/bin/bash
RUNS=25
USE_VAL_DATA=false
RUN_PARALLEL_SIMULATION=true
export PORT="${PORT:-5000}"
export PROJECT_NAME="${PROJECT_NAME:-test}"
export RESAMPLE=true
export TOTAL_ITERATIONS=50
TRAINER_TYPE=bayesian


echo $PROJECT_NAME
sleep 10
for DATASET in  tax omdb hospital airport 
do
    for MAX_DIRTY_PROP in 0.3 #0.05 0.1
    do
        echo "Dumping data.."
        python dump_processed_data.py --dataset $DATASET --max-clean-num 1000 --max-dirty-prop $MAX_DIRTY_PROP

        # dump converged global trainer model
        ./dump_converged_model.sh $DATASET

        (cd learner && gunicorn -w 25 --bind 0.0.0.0:$PORT --log-level info --timeout 240 api:app)&
        
        sleep 30
        cd ./trainer

        # for TRAINER_TYPE in full-oracle learning-oracle bayesian
        for TRAINER_PRIOR_TYPE in random # all # data-estimate uniform-0.1 uniform-0.9 random
        do
            for LEARNER_PRIOR_TYPE in random # all #data-estimate uniform-0.1 uniform-0.9 random
            do
                for SAMPLING_TYPE  in all #RANDOM ACTIVELR STOCHASTICBR STOCHASTICUS
                do
                    echo "Running simulation for $DATASET $TRAINER_TYPE $SAMPLING_TYPE $RUNS $USE_VAL_DATA $TRAINER_PRIOR_TYPE $LEARNER_PRIOR_TYPE"
                    python simulate.py $DATASET $TRAINER_TYPE $SAMPLING_TYPE $RUNS $USE_VAL_DATA $TRAINER_PRIOR_TYPE $LEARNER_PRIOR_TYPE $RUN_PARALLEL_SIMULATION
                done
                
            done
        done
        
        cd ..
        pkill -f gunicorn
        sleep 10
    done
done
sleep 10
python plot_figures.py