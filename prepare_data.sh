#!/bin/bash

python dump_hypothesis_space_info.py --data-path ./data/raw-data/clean_tax.csv --dataset-name tax --rel-fields zip city state haschild childexemp maritalstatus singleexemp --noise-ratio 0.3
python dump_hypothesis_space_info.py --data-path ./data/raw-data/clean_hospital.csv --dataset-name hospital --rel-fields zip city state phone measurecode stateavg --noise-ratio 0.3

python initial_compute_fd_tuple_info.py --dataset hospital
python initial_compute_fd_tuple_info.py --dataset tax
