# Exploratory Training: When Annotators Learn About Data

## Setup environment

1. Change the current directory to the project directory  
   `cd <project_dir>`
2. Create a new virtual environment.  
   **Note:** Requires python version >= 3.7  
   `virtualenv --python=python3 .venv`
3. Activate virtual environment  
   `source .venv/bin/activate`
4. Install requirements  
   `pip install -r learner/requirements.txt`

## Process and prepare data

**Note:** This requires the raw `*.csv` for each scenario inside `<project_dir>/data/raw-data` folder.

1. Initial parallel precomputation: prepare hypothesis space, process, sample and prepare data for simulation  
   `sh ./prepare_data.sh`

2. Run simulation and plot the figures  
   `sh ./run_simulation.sh`

The results on the learner side are stored in `<project_dir>/learner/store` whereas on the trainer side are stored in `<project_dir>/trainer/trainer-store`. The figures are store in the directory `<project_dir>/figures`.
