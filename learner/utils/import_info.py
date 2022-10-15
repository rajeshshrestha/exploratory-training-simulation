from ast import Raise
import json
import os
import pickle
import logging
from flask import request
from flask_restful import Resource
from rich.console import Console
from .initialize_variables import scenarios, processed_dfs, validation_indices_dict, models_dict
from .helper import StudyMetric, FDMeta, initialPrior
import random

console = Console()
logger = logging.getLogger(__file__)


class Import(Resource):
    def get(self):
        return {'msg': '[SUCCESS] /duo/api/import is live!'}

    def post(self):
        # Initialize a new project
        # projects = [('0x' + d) for d in os.listdir('./store') if
        #             os.path.isdir(os.path.join('./store/', d))]
        # if len(projects) == 0:
        #     new_project_id = '{:08x}'.format(1)
        # else:
        #     project_ids = [int(d, 0) for d in projects]
        #     new_project_id = '{:08x}'.format(max(project_ids) + 1)

        # Read the scenario number and initialize the scenario accordingly
        scenario_id = request.form.get('scenario_id')
        email = request.form.get('email')
        initial_user_h = request.form.get('initial_fd')
        fd_comment = request.form.get('fd_comment')
        skip_user = (request.form.get('skip_user').lower() == "true")
        sampling_method = request.form.get('sampling_method')
        trainer_type = request.form.get('trainer_type')
        use_val_data = (request.form.get('use_val_data').lower() == "true")
        learner_prior_type = request.form.get('learner_prior_type')
        trainer_prior_type = request.form.get('trainer_prior_type')

        if scenario_id is None or email is None:
            scenario_id = json.loads(request.data)['scenario_id']
            email = json.loads(request.data)['email']
            initial_user_h = json.loads(request.data)['initial_fd']
            fd_comment = json.loads(request.data)['fd_comment']
            skip_user = False if 'skip_user' not in json.loads(
                request.data).keys() else (json.loads(request.data)['skip_user'].lower() == 'true')
            sampling_method = json.loads(request.data)['sampling_type']
            trainer_type = json.loads(request.data)['trainer_type']
            use_val_data = (json.loads(request.data)[
                            'use_val_data'].lower() == 'true')
            learner_prior_type = json.loads(request.data)['learner_prior_type']
            trainer_prior_type = json.loads(request.data)['trainer_prior_type']

        logger.info(initial_user_h)

        project_base_dir = f"dataset={scenario_id}/use_val_data={use_val_data}/" +\
            f"dirty-proportion={round(models_dict[scenario_id]['dirty_proportion'],2)}/" +\
            f"trainer-prior-type={trainer_prior_type}-learner-prior-type=" +\
            f"{learner_prior_type}"

        new_project_id = project_base_dir+"/" +\
            trainer_type + "_" + sampling_method + \
            "_"+str(random.randint(1, 1e15))
        new_project_dir = './store/' + new_project_id

        # Save the new project
        try:
            os.makedirs(f'./store/{project_base_dir}',
                        exist_ok=True)
            os.mkdir(new_project_dir)
        except OSError:
            returned_data = {
                'msg': '[ERROR] Unable to create a directory for this project.'
            }
            logger.info(returned_data)
            response = json.dumps(returned_data)
            return response, 500, {'Access-Control-Allow-Origin': '*'}

        logger.info('*** Project initialized ***')

        if not skip_user:
            # Get the user from the users list
            try:
                users = pickle.load(open('./study-utils/users.pk', 'rb'))
            except Exception as e:
                return {'msg': '[ERROR] users does not exist'}, 400, {
                    'Access-Control-Allow-Origin': '*'}

            # Save the user's questionnaire responses
            if email not in users.keys():
                return {
                    'msg': '[ERROR] no user exists with this email'}, 400, {
                    'Access-Control-Allow-Origin': '*'}

            user = users[email]
            user.scenarios = user.scenarios[1:]
            user.runs.append(new_project_id)
            users[email] = user

            # Save the users object updates
            pickle.dump(users, open('./study-utils/users.pk', 'wb'))

        project_info = {
            'email': email,
            'scenario_id': scenario_id,
            'sampling_method': sampling_method,
            'trainer_type': trainer_type,
            'use_val_data': use_val_data,
            'trainer_prior_type': trainer_prior_type,
            'learner_prior_type': learner_prior_type
        }

        with open(new_project_dir + '/project_info.json', 'w') as f:
            json.dump(project_info, f, indent=4)

        logger.info('*** Project info saved ***')

        data = processed_dfs[scenario_id]
        header = [col for col in data.columns if col != 'is_clean']

        # Initialize the iteration counter
        current_iter = 0

        # Initialize metadata objects
        interaction_metadata = dict()
        interaction_metadata['header'] = header
        interaction_metadata['user_hypothesis_history'] = [
            StudyMetric(iter_num=current_iter,
                        value=[initial_user_h, fd_comment],
                        elapsed_time=0)]
        interaction_metadata['feedback_history'] = dict()
        interaction_metadata['feedback_recent'] = dict()
        interaction_metadata['sample_history'] = list()
        for idx in data.index:
            interaction_metadata['feedback_history'][idx] = dict()
            interaction_metadata['feedback_recent'][idx] = dict()
            for col in header:
                interaction_metadata['feedback_history'][idx][
                    col] = list()
                interaction_metadata['feedback_recent'][idx][col] = False

        current_iter += 1

        '''Initialize the learner model'''
        fd_metadata = get_initial_fd_metadata(scenario_id=scenario_id,
                                              prior_type=learner_prior_type)

        study_metrics = dict()
        # study_metrics['iter_err_precision'] = list()
        # study_metrics['iter_err_recall'] = list()
        # study_metrics['iter_err_f1'] = list()
        # study_metrics['all_err_precision'] = list()
        # study_metrics['all_err_recall'] = list()
        # study_metrics['all_err_f1'] = list()
        study_metrics['iter_accuracy'] = list()
        study_metrics['iter_recall'] = list()
        study_metrics['iter_precision'] = list()
        study_metrics['iter_f1'] = list()
        study_metrics['iter_mae_ground_model_error'] = list()
        study_metrics['iter_mae_trainer_model_error'] = list()
        study_metrics['elapsed_time'] = list()
        json.dump(study_metrics,
                  open(new_project_dir + '/study_metrics.json', 'w'))

        # Initialize tuple metadata and value metadata objects
        tuple_weights = dict()
        for idx in range(0, len(data)):
            # Tuple metadata
            tuple_weights[idx] = 1 / len(data)

        pickle.dump(interaction_metadata,
                    open(new_project_dir + '/interaction_metadata.pk', 'wb'))
        pickle.dump(tuple_weights,
                    open(new_project_dir + '/tuple_weights.pk', 'wb'))
        pickle.dump(fd_metadata,
                    open(new_project_dir + '/fd_metadata.pk', 'wb'))
        pickle.dump(current_iter,
                    open(new_project_dir + '/current_iter.pk', 'wb'))

        if use_val_data:
            logger.info(
                "Using all data including validation data in interaction")
            total_indices = set(data.index)
        else:
            logger.info("Using data except val data for training")
            total_indices = set(data.index) - \
                validation_indices_dict[scenario_id]
        pickle.dump(total_indices,
                    open(new_project_dir + '/unserved_indices.pk', 'wb'))

        logger.info('*** Metadata and objects initialized and saved ***')

        # Return information to the user
        response = {
            'project_id': new_project_id
        }
        return response, 201, {'Access-Control-Allow-Origin': '*'}


def get_initial_fd_metadata(scenario_id, prior_type):

    scenario = scenarios[scenario_id]

    # Initialize hypothesis parameters
    fd_metadata = dict()

    variance = 0.0025  # hyperparameter
    logger.info(f"Initializing learner prior model with variance: {variance}")
    if prior_type in ['uniform-0.1',
                      'uniform-0.5',
                      'uniform-0.9']:
        mu = float(prior_type.split("-")[1])
        logger.info(f"mu: {mu}")

        for h in scenario['hypothesis_space']:

            # Calculate alpha and beta
            alpha, beta = initialPrior(mu, variance)

            # Initialize the FD metadata object
            fd_m = FDMeta(
                fd=h,
                a=alpha,
                b=beta,
            )

            logger.info('iter: 0'),
            logger.info(f'alpha: {fd_m.alpha}')
            logger.info(f'beta: {fd_m.beta}')

            fd_metadata[h] = fd_m

    elif prior_type == 'random':
        for h in scenario['hypothesis_space']:
            mu = random.uniform(0, 1)
            # Calculate alpha and beta
            alpha, beta = initialPrior(mu, variance)

            # Initialize the FD metadata object
            fd_m = FDMeta(
                fd=h,
                a=alpha,
                b=beta,
            )

            logger.info('iter: 0'),
            logger.info(f"mu: {mu}")
            logger.info(f'alpha: {fd_m.alpha}')
            logger.info(f'beta: {fd_m.beta}')

            fd_metadata[h] = fd_m

    elif prior_type == 'data-estimate':
        for h in scenario['hypothesis_space']:
            mu = compute_hyp_conf_in_data(fd=h,
                                          scenario_id=scenario_id)
            # Calculate alpha and beta
            alpha, beta = initialPrior(mu, variance)

            # Initialize the FD metadata object
            fd_m = FDMeta(
                fd=h,
                a=alpha,
                b=beta,
            )

            logger.info('iter: 0'),
            logger.info(f"mu: {mu}")
            logger.info(f'alpha: {fd_m.alpha}')
            logger.info(f'beta: {fd_m.beta}')

            fd_metadata[h] = fd_m
    else:
        raise Exception(
            f"Invalid prior type: {prior_type} specified for learner!!!")

    return fd_metadata


def compute_hyp_conf_in_data(fd, scenario_id):
    scenario = scenarios[scenario_id]
    support_num = len(scenario['hypothesis_space'][fd]['support_pairs'])
    violation_num = len(scenario['hypothesis_space'][fd]['support_pairs'])

    return support_num/(support_num+violation_num+1e-7)
