import os

TOTAL_SCENARIOS = 1
TOTAL_ITERATIONS = int(os.getenv("TOTAL_ITERATIONS",15)) #30
RESAMPLE = (os.getenv("RESAMPLE", "false").lower() == "true")
SAMPLE_SIZE = 10
MODEL_FDS_TOP_K = 10
# ACCURACY_ESTIMATION_SAMPLE_NUM = 1000 #directly sampled from dataset during dataprocessing for now
ACTIVE_LEARNING_CANDIDATE_INDICES_NUM = 200# 400
STOCHASTIC_BEST_RESPONSE_CANDIDATE_INDICES_NUM = 200 #400
STOCHASTIC_ACTIVE_LEARNING_CANDIDATE_INDICES_NUM = 200 #400

STOCHASTIC_BEST_RESPONSE_GAMMA = 0.5
STOCHASTIC_UNCERTAINTY_SAMPLING_GAMMA = 0.5

LEARNER_PRIOR_VARIANCE=0.0025

project_name = os.getenv("PROJECT_NAME", None)
STORE_BASE_PATH = os.getenv("STORE_BASE_PATH", None)
if STORE_BASE_PATH is None:
    if project_name is None:
        STORE_BASE_PATH = "./store"
    else:
        STORE_BASE_PATH= os.path.join("./run-data", project_name, "store")
        
    os.makedirs(STORE_BASE_PATH, exist_ok=True)
    
