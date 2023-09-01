#!/bin/bash
export TOTAL_ITERATIONS=200
export RESAMPLE=true
export PROJECT_NAME="global_converging_trainer_model"
export PORT="$(($PORT+1))"
(cd learner && gunicorn -w 10 --bind 0.0.0.0:$PORT --log-level info --timeout 240 api:app)&
sleep 30

cd trainer
echo "Computing and dumping Global converged model...."
python global_converged_model_dumper.py  $1

pkill -f gunicorn
sleep 30
echo "Finished Dumping Converged Model"
