import json
import pickle
import time

import pandas as pd
from flask import request
from flask_restful import Resource
from rich.console import Console

from .env_variables import SAMPLING_METHOD, RESAMPLE
from .helper import buildSample

console = Console()


# Get the first sample for a scenario interaction
class Sample(Resource):
    def get(self):
        return {'msg': '[SUCCESS] /duo/api/sample is live!'}

    def post(self):
        # Get the project ID
        project_id = request.form.get('project_id')
        if project_id is None:
            # print(request.data)
            project_id = json.loads(request.data)['project_id']
        sample_size = 10
        with open('./store/' + project_id + '/project_info.json') as f:
            project_info = json.load(f)

        # Calculate the start time of the interaction
        start_time = time.time()
        pickle.dump(start_time,
                    open('./store/' + project_id + '/start_time.p', 'wb'))

        print('*** Project info loaded ***')

        # Get data
        data = pd.read_csv(project_info['scenario']['dirty_dataset'],
                           keep_default_na=False)
        current_iter = pickle.load(
            open('./store/' + project_id + '/current_iter.p', 'rb'))
        X = pickle.load(open('./store/' + project_id + '/X.p',
                             'rb'))  # list of true violation pairs (vio pairs for target FD)

        # Build sample
        s_out, sample_X = buildSample(data, sample_size, project_id,
                                      sampling_method=SAMPLING_METHOD, resample=RESAMPLE)
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
        pickle.dump(sample_X,
                    open('./store/' + project_id + '/current_X.p', 'wb'))

        # Add ID to s_out (for use on frontend)
        s_out.insert(0, 'id', s_out.index, True)

        # Build initial feedback map for frontend
        feedback = list()
        for idx in s_out.index:
            for col in s_out.columns:
                feedback.append({
                    'row': idx,
                    'col': col,
                    'marked': False
                })

        print('*** Feedback object created ***')

        # Return information to the user
        response = {
            'sample': s_out.to_json(orient='index'),
            'X': [list(v) for v in sample_X],
            'feedback': json.dumps(feedback),
            'msg': '[SUCCESS] Successfully built sample.'
        }
        return response, 200, {'Access-Control-Allow-Origin': '*'}
