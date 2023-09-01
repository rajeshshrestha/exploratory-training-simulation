import os
from simulate import run
import json
import sys

def dump_converged_model(scenario):
    model = run(scenario_id=scenario,
        sampling_method="RANDOM",
        trainer_type="bayesian",
        use_val_data=False,
        trainer_prior_type="random",
        learner_prior_type="random",
        is_global= True)
    
    os.makedirs("../converged_models", exist_ok=True)
    with open(f"../converged_models/converged_global_trainer_{scenario}.json", "w") as fp:
        json.dump(model, fp)



if __name__ == "__main__":
    scenario = sys.argv[1] if sys.argv[1] is not None else 'omdb'
    dump_converged_model(scenario)

