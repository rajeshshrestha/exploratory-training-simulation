import json
import pickle
import time

import pandas as pd
from flask import request
from flask_restful import Resource
from rich.console import Console

from .env_variables import SAMPLING_METHOD, TOTAL_ITERATIONS, RESAMPLE
from .helper import buildSample, interpretFeedback, StudyMetric, recordFeedback

console = Console()


class Feedback(Resource):
    def get(self):
        return {'msg': '[SUCCESS] /duo/api/feedback is live!'}

    def post(self):
        # Get the project ID for the interaction and the user's feedback object
        project_id = request.form.get('project_id')
        current_user_h = request.form.get('current_user_h')
        user_h_comment = request.form.get('user_h_comment')
        if project_id is None:
            req = json.loads(request.data)
            project_id = req['project_id']
            feedback_dict = req['feedback']
            current_user_h = req['current_user_h']
            user_h_comment = req['user_h_comment']
        else:
            feedback_dict = json.loads(request.form.get('feedback'))

        print(project_id)
        console.log(current_user_h)

        feedback = pd.DataFrame.from_dict(feedback_dict, orient='index')
        sample_size = 10

        print('*** Necessary objects loaded ***')

        # Get the current iteration count and current time
        current_iter = pickle.load(
            open('./store/' + project_id + '/current_iter.p', 'rb'))
        print(current_iter)
        current_time = time.time()

        curr_sample_X = pickle.load(
            open('./store/' + project_id + '/current_X.p', 'rb'))
        X = pickle.load(open('./store/' + project_id + '/X.p', 'rb'))

        print('*** Iteration counter updated ***')

        # Get the project info
        with open('./store/' + project_id + '/project_info.json', 'r') as f:
            project_info = json.load(f)

        print('*** Project info loaded ***')

        # Load the dataset
        data = pd.read_csv(project_info['scenario']['dirty_dataset'],
                           keep_default_na=False)

        print('*** Loaded dirty dataset ***')

        # Record the user's feedback and analyze it
        s_in = data.iloc[feedback.index]
        print('*** Extracted sample from dataset ***')
        recordFeedback(data, feedback_dict, curr_sample_X, project_id,
                       current_iter, current_time)
        target_fd = project_info['scenario'][
            'target_fd']  # NOTE: For current sims only
        interpretFeedback(s_in, feedback, project_id,
                          current_iter, current_time)

        # Build a new sample
        current_iter += 1
        s_out, new_sample_X = buildSample(data, sample_size,
                                          project_id,
                                          sampling_method=SAMPLING_METHOD,
                                          resample=RESAMPLE)
        s_index = s_out.index

        # open file containing the indices of unserved tuples, update it and dump
        unserved_indices = pickle.load(
            open('./store/' + project_id + '/unserved_indices.p', 'rb'))
        unserved_indices = list(set(unserved_indices) - set(s_index))
        pickle.dump(unserved_indices,
                    open('./store/' + project_id + '/unserved_indices.p',
                         'wb'))

        pickle.dump(s_index,
                    open('./store/' + project_id + '/current_sample.p', 'wb'))
        pickle.dump(new_sample_X,
                    open('./store/' + project_id + '/current_X.p', 'wb'))

        s_out.insert(0, 'id', s_out.index, True)
        print(s_out.index)

        # Build feedback map for front-end
        start_time = pickle.load(
            open('./store/' + project_id + '/start_time.p', 'rb'))
        elapsed_time = current_time - start_time
        feedback = list()
        interaction_metadata = pickle.load(
            open('./store/' + project_id + '/interaction_metadata.p', 'rb'))
        interaction_metadata['user_hypothesis_history'].append(
            StudyMetric(iter_num=current_iter - 1,
                        value=[current_user_h, user_h_comment],
                        elapsed_time=elapsed_time))  # current iter - 1 because it's for the prev iter (i.e. before incrementing current_iter)
        pickle.dump(interaction_metadata,
                    open('./store/' + project_id + '/interaction_metadata.p',
                         'wb'))

        for idx in s_out.index:
            for col in s_out.columns:
                feedback.append({
                    'row': idx,
                    'col': col,
                    'marked': False if col == 'id' else (bool(
                        interaction_metadata['feedback_history'][int(idx)][
                            col][-1].marked) if len(
                        interaction_metadata['feedback_history'][int(idx)][
                            col]) > 0 else False)
                })

        print('*** Feedback object created ***')

        # Check if the scenario is done
        # if current_iter <= 5:
        #     terminate = False
        # else:
        #     terminate = helpers.checkForTermination(project_id)
        if current_iter > TOTAL_ITERATIONS or ((len(unserved_indices) == 0) and (not RESAMPLE)):
            msg = '[DONE]'
        else:
            msg = '[SUCCESS]: Saved feedback and built new sample.'

        # Save object updates
        pickle.dump(current_iter,
                    open('./store/' + project_id + '/current_iter.p', 'wb'))
        with open('./store/' + project_id + '/project_info.json', 'w') as f:
            json.dump(project_info, f)

        # Return information to the user
        response = {
            'sample': s_out.to_json(orient='index'),
            'X': [list(v) for v in new_sample_X],
            'feedback': json.dumps(feedback),
            'msg': msg
        }
        return response, 200, {'Access-Control-Allow-Origin': '*'}
