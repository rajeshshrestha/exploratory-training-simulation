import sys

from functools import partial
from multiprocessing import Pool
import os

from utils.interact_learner import initialize_learner, get_initial_sample
from utils.interact_learner import send_feedback
from user_models.trainer import TrainerModel


def run(scenario_id, trainer_type, sampling_method):

    # Initialize the learner
    project_id, _ = initialize_learner(scenario_id=scenario_id,
                                       sampling_method=sampling_method,
                                       trainer_type=trainer_type)

    # Get the first batch of sample from the learner
    data, columns, feedback = get_initial_sample(project_id=project_id)

    # initialize the trainer
    trainer = TrainerModel(trainer_type=trainer_type,
                           scenario_id=scenario_id,
                           project_id=project_id,
                           columns=columns)

    msg = ''
    iter_num = 0

    # Begin iterations and continue till done
    while msg != '[DONE]' and (len(data) > 0):
        iter_num += 1

        feedback_dict = trainer.get_feedback_dict(data, feedback)

        '''Get Model FD conf dict for metric computation on learner side'''
        model_dict = trainer.get_model()
        data, feedback, msg = send_feedback(project_id=project_id,
                                            feedback_dict=feedback_dict,
                                            model_dict=model_dict)


if __name__ == '__main__':
    '''
    Arg 1: Scenarios
    Arg 2: Bayesian Type
            Values: ["oracle", "informed", "uninformed"]
    Arg 3: Decision Type
            Values: ["coinflip", "threshold]
    Arg 4: Number of runs
    Arg 5: Whether precision or recall
    '''

    # Scenario
    scenario_id = sys.argv[1] if sys.argv[1] is not None else 'omdb'
    trainer_type = sys.argv[
        2]
    sampling_method = sys.argv[3]
    num_runs = int(sys.argv[4])  # How many runs of this simulation to do

    assert trainer_type in ['full-oracle',
                            'learning-oracle',
                            'uninformed-bayesian'],\
        "Invalid trainer type passed!!!"
    assert sampling_method in ['RANDOM',
                               'ACTIVELR',
                               'STOCHASTICBR',
                               'STOCHASTICUS'],\
        "Invalid sampling method passed!!!"

    cpu_num = os.cpu_count()

    with Pool(10) as p:
        p.map(partial(run,
                      trainer_type=trainer_type,
                      sampling_method=sampling_method),
              [scenario_id for i in range(num_runs)])
