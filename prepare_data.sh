#!/bin/bash

python dump_hypothesis_space_info.py --data-path ./data/raw-data/clean_tax.csv --dataset-name tax --rel-fields zip city state haschild childexemp maritalstatus singleexemp
python dump_hypothesis_space_info.py --data-path ./data/raw-data/omdb-clean-full.csv --dataset-name omdb --rel-fields title director rating year genre type
python dump_hypothesis_space_info.py --data-path ./data/raw-data/airport-clean-full.csv --dataset-name airport --rel-fields owner sitenumber facilityname manager type ownership locationid
python dump_hypothesis_space_info.py --data-path ./data/raw-data/clean_hospital.csv --dataset-name hospital --rel-fields zip city state measurecode stateavg hospitalname hospitaltype


python initial_compute_fd_tuple_info.py --dataset airport
python initial_compute_fd_tuple_info.py --dataset omdb
python initial_compute_fd_tuple_info.py --dataset hospital
python initial_compute_fd_tuple_info.py --dataset tax