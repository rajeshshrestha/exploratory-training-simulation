# Exploratory Training: When Annotators Learn About Data

## Setup environment
1. Change the current directory to the project directory     
`cd <project_dir>`   
2. Create a new virtual environment    
`virtualenv --python=python3 .venv`   
3. Activate virtual environment    
`source .venv/bin/activate`
4. Install requirements    
`pip install -r learner/requirements.txt`
## Process and prepare data
__Note:__ This requires the `scenarios.json` which consists the information on hypothesis space and the raw-data set at `project_dir/data/raw-data`
1. Initial parallel precomputation and processing data  
`python initial_compute_fd_tuple_info.py --dataset <dataset>`    
Here, <dataset> can be either _omdb_ or _airport_.

2. Process, sample and prepare data for simulation   
`python dump_processed_data.py --max-clean-num <max-clean-num> --max-dirty-prop <violation ratio> --dataset <dataset>`    

3. Run simulation    
`sh ./run_simulation.sh`

The results on the learner side are stored in `learner/store` whereas on the trainer side are stored in `trainer/trainer-store`.

## Plot the results
` python plot_figures.py`    
The figures are store in the directory `<project_dir>/figures`.