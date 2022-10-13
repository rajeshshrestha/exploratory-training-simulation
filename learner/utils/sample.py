import json
import pickle
import time
import logging

from flask import request
from flask_restful import Resource
from rich.console import Console

from .env_variables import RESAMPLE, SAMPLE_SIZE
from .helper import buildSample
from .initialize_variables import processed_dfs

logger = logging.getLogger(__file__)
console = Console()


# Get the first sample for a scenario interaction
class Sample(Resource):
    def get(self):
        return {'msg': '[SUCCESS] /duo/api/sample is live!'}

    def post(self):
        # Get the project ID
        project_id = request.form.get('project_id')
        if project_id is None:
            project_id = json.loads(request.data)['project_id']

        with open('./store/' + project_id + '/project_info.json') as f:
            project_info = json.load(f)
            sampling_method = project_info['sampling_method']
        scenario_id = project_info['scenario_id']

        # Calculate the start time of the interaction
        start_time = time.time()
        pickle.dump(start_time,
                    open('./store/' + project_id + '/start_time.pk', 'wb'))

        logger.info('*** Project info loaded ***')

        data = processed_dfs[scenario_id]

        # Build sample
        s_out = buildSample(data, SAMPLE_SIZE, project_id,
                            sampling_method=sampling_method,
                            resample=RESAMPLE)
        s_index = s_out.index

        # open file containing the indices of unserved tuples, update it and dump
        unserved_indices = pickle.load(
            open('./store/' + project_id + '/unserved_indices.pk', 'rb'))
        unserved_indices = (set(unserved_indices) - set(s_index))
        pickle.dump(unserved_indices,
                    open('./store/' + project_id + '/unserved_indices.pk',
                         'wb'))

        pickle.dump(s_index,
                    open('./store/' + project_id + '/current_sample.pk', 'wb'))

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

        logger.info('*** Feedback object created ***')

        # Return information to the user
        response = {
            'sample': s_out.to_json(orient='index'),
            'feedback': json.dumps(feedback),
            'msg': '[SUCCESS] Successfully built sample.'
        }
        return response, 200, {'Access-Control-Allow-Origin': '*'}
