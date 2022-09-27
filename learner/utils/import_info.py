import json
import os
import pickle
from pprint import pprint

import pandas as pd
from flask import request
from flask_restful import Resource
from rich.console import Console

from .helper import StudyMetric, FDMeta, initialPrior
from .env_variables import TOTAL_SCENARIOS

console = Console()


class Import(Resource):
    def get(self):
        return {'msg': '[SUCCESS] /duo/api/import is live!'}

    def post(self):
        # Initialize a new project
        projects = [('0x' + d) for d in os.listdir('./store') if
                    os.path.isdir(os.path.join('./store/', d))]
        if len(projects) == 0:
            new_project_id = '{:08x}'.format(1)
        else:
            project_ids = [int(d, 0) for d in projects]
            new_project_id = '{:08x}'.format(max(project_ids) + 1)
        new_project_dir = './store/' + new_project_id

        # Save the new project
        try:
            os.mkdir(new_project_dir)
        except OSError:
            returned_data = {
                'msg': '[ERROR] Unable to create a directory for this project.'
            }
            pprint(returned_data)
            response = json.dumps(returned_data)
            return response, 500, {'Access-Control-Allow-Origin': '*'}

        print('*** Project initialized ***')

        # Read the scenario number and initialize the scenario accordingly
        scenario_id = request.form.get('scenario_id')
        email = request.form.get('email')
        initial_user_h = request.form.get('initial_fd')
        fd_comment = request.form.get('fd_comment')
        skip_user = request.form.get('skip_user')
        violation_ratio = request.form.get('violation_ratio')
        if scenario_id is None or email is None:
            scenario_id = json.loads(request.data)['scenario_id']
            email = json.loads(request.data)['email']
            initial_user_h = json.loads(request.data)['initial_fd']
            fd_comment = json.loads(request.data)['fd_comment']
            skip_user = False if 'skip_user' not in json.loads(
                request.data).keys() else json.loads(request.data)['skip_user']
            violation_ratio = None if 'violation_ratio' not in json.loads(
                request.data).keys() else json.loads(request.data)[
                'violation_ratio']
        print(scenario_id)
        console.log(initial_user_h)

        if not skip_user:
            # Get the user from the users list
            try:
                users = pickle.load(open('./study-utils/users.p', 'rb'))
            except:
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
            user_interaction_number = TOTAL_SCENARIOS - len(user.scenarios)
            users[email] = user

            # Save the users object updates
            pickle.dump(users, open('./study-utils/users.p', 'wb'))

        else:
            user_interaction_number = 1 if violation_ratio == 'close' else 5

        with open('../scenarios.json', 'r') as f:
            scenarios_list = json.load(f)
        scenario = scenarios_list[scenario_id]
        if user_interaction_number <= 3:
            target_h_sample_ratio = 0.2
            alt_h_sample_ratio = 0.6
        else:
            target_h_sample_ratio = 0.3
            alt_h_sample_ratio = 0.45
        scenario['target_h_sample_ratio'] = target_h_sample_ratio
        scenario['alt_h_sample_ratio'] = alt_h_sample_ratio
        target_fd = scenario['target_fd']
        project_info = {
            'email': email,
            'scenario_id': scenario_id,
            'scenario': scenario
        }

        with open(new_project_dir + '/project_info.json', 'w') as f:
            json.dump(project_info, f, indent=4)

        print('*** Project info saved ***')

        data = pd.read_csv(scenario['dirty_dataset'])
        header = [col for col in data.columns]
        # random.shuffle(header)

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
        interaction_metadata['sample_history'] = list()
        for idx in data.index:
            interaction_metadata['feedback_history'][int(idx)] = dict()
            for col in header:
                interaction_metadata['feedback_history'][int(idx)][
                    col] = list()

        # Initialize hypothesis parameters
        fd_metadata = dict()
        h_space = scenario['hypothesis_space']
        for h in h_space:

            # Calculate the mean and variance
            h['vio_pairs'] = set(tuple(vp) for vp in h['vio_pairs'])

            # todo: uncheck this previous
            # mu = h['conf']      # h['conf'] = # tuples that satisfy FD / # tuples total
            mu = 0.1
            if mu == 1:
                mu = 0.99999
            variance = 0.0025  # hyperparameter

            # Calculate alpha and beta
            alpha, beta = initialPrior(mu, variance)

            # Initialize the FD metadata object
            fd_m = FDMeta(
                fd=h['cfd'],
                a=alpha,
                b=beta,
                support=h['support'],
                vios=h['vios'],
                vio_pairs=h['vio_pairs'],
            )

            print('iter: 0'),
            print('alpha:', fd_m.alpha)
            print('beta:', fd_m.beta)
            print('conf:', h['conf'])

            fd_metadata[h['cfd']] = fd_m

        current_iter += 1

        study_metrics = dict()
        study_metrics['iter_err_precision'] = list()
        study_metrics['iter_err_recall'] = list()
        study_metrics['iter_err_f1'] = list()
        study_metrics['all_err_precision'] = list()
        study_metrics['all_err_recall'] = list()
        study_metrics['all_err_f1'] = list()
        pickle.dump(study_metrics,
                    open(new_project_dir + '/study_metrics.p', 'wb'))

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
        pickle.dump(fd_metadata[target_fd].vio_pairs,
                    open(new_project_dir + '/X.p', 'wb'))

        total_indices = set()
        for fd in project_info['scenario']['hypothesis_space']:
            total_indices |= set(fd['support'])
        pickle.dump(total_indices,
                    open(new_project_dir + '/unserved_indices.p', 'wb'))

        print('*** Metadata and objects initialized and saved ***')

        # Return information to the user
        response = {
            'project_id': new_project_id,
            'description': scenario['description']
        }
        return response, 201, {'Access-Control-Allow-Origin': '*'}
