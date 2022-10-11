import json
import os
import pickle
import logging
import pandas as pd
from flask import request
from flask_restful import Resource
from rich.console import Console
from .initialize_variables import scenarios, processed_dfs
from .helper import StudyMetric, FDMeta, initialPrior
import random
from .env_variables import SAMPLING_METHOD

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

        new_project_id = SAMPLING_METHOD+"_"+str(random.randint(1, 1e15))
        new_project_dir = './store/' + new_project_id

        # Save the new project
        try:
            os.mkdir(new_project_dir)
        except OSError:
            returned_data = {
                'msg': '[ERROR] Unable to create a directory for this project.'
            }
            logger.info(returned_data)
            response = json.dumps(returned_data)
            return response, 500, {'Access-Control-Allow-Origin': '*'}

        logger.info('*** Project initialized ***')

        # Read the scenario number and initialize the scenario accordingly
        scenario_id = request.form.get('scenario_id')
        email = request.form.get('email')
        initial_user_h = request.form.get('initial_fd')
        fd_comment = request.form.get('fd_comment')
        skip_user = request.form.get('skip_user')
        if scenario_id is None or email is None:
            scenario_id = json.loads(request.data)['scenario_id']
            email = json.loads(request.data)['email']
            initial_user_h = json.loads(request.data)['initial_fd']
            fd_comment = json.loads(request.data)['fd_comment']
            skip_user = False if 'skip_user' not in json.loads(
                request.data).keys() else json.loads(request.data)['skip_user']

        logger.info(initial_user_h)

        if not skip_user:
            # Get the user from the users list
            try:
                users = pickle.load(open('./study-utils/users.p', 'rb'))
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
            pickle.dump(users, open('./study-utils/users.p', 'wb'))

        project_info = {
            'email': email,
            'scenario_id': scenario_id,
        }

        with open(new_project_dir + '/project_info.json', 'w') as f:
            json.dump(project_info, f, indent=4)

        logger.info('*** Project info saved ***')

        data = processed_dfs[scenario_id]
        header = [col for col in data.columns if col != 'is_clean']
        scenario = scenarios[scenario_id]

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

        # Initialize hypothesis parameters
        fd_metadata = dict()
        for h in scenario['hypothesis_space']:
            mu = 0.1
            if mu == 1:
                mu = 0.99999
            variance = 0.0025  # hyperparameter

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

        current_iter += 1

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
        study_metrics['iter_mae_model_error'] =  list()
        study_metrics['elapsed_time'] = list()
        json.dump(study_metrics,
                  open(new_project_dir + '/study_metrics.json', 'w'))

        # Initialize tuple metadata and value metadata objects
        tuple_weights = dict()
        for idx in range(0, len(data)):
            # Tuple metadata
            tuple_weights[idx] = 1 / len(data)

        pickle.dump(interaction_metadata,
                    open(new_project_dir + '/interaction_metadata.p', 'wb'))
        pickle.dump(tuple_weights,
                    open(new_project_dir + '/tuple_weights.p', 'wb'))
        pickle.dump(fd_metadata,
                    open(new_project_dir + '/fd_metadata.p', 'wb'))
        pickle.dump(current_iter,
                    open(new_project_dir + '/current_iter.p', 'wb'))

        total_indices = set(data.index)
        pickle.dump(total_indices,
                    open(new_project_dir + '/unserved_indices.p', 'wb'))

        logger.info('*** Metadata and objects initialized and saved ***')

        # Return information to the user
        response = {
            'project_id': new_project_id
        }
        return response, 201, {'Access-Control-Allow-Origin': '*'}
