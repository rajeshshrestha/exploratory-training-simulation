import json
import pickle
import time
import logging
import os

import pandas as pd
from flask import request
from flask_restful import Resource
from rich.console import Console
from copy import deepcopy

from .env_variables import TOTAL_ITERATIONS, RESAMPLE, SAMPLE_SIZE
from .helper import buildSample, interpretFeedback, StudyMetric, recordFeedback
from .initialize_variables import processed_dfs, validation_indices_dict
from .metrics import compute_metrics_using_converged_model
from .env_variables import MODEL_FDS_TOP_K, STORE_BASE_PATH

console = Console()
logger = logging.getLogger(__file__)


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
            trainer_model = req['trainer_model']
        else:
            feedback_dict = json.loads(request.form.get('feedback'))
            trainer_model = json.loads(request.form.get('trainer_model'))

        logger.info(project_id)
        logger.info(current_user_h)

        feedback = pd.DataFrame.from_dict(feedback_dict, orient='index')

        logger.info('*** Necessary objects loaded ***')

        # Get the current iteration count and current time
        current_iter = pickle.load(
            open(f'{STORE_BASE_PATH}/' + project_id + '/current_iter.pk', 'rb'))
        logger.info(current_iter)
        current_time = time.time()

        logger.info('*** Iteration counter updated ***')

        # Get the project info
        with open(f'{STORE_BASE_PATH}/' + project_id + '/project_info.json', 'r') as f:
            project_info = json.load(f)
            sampling_method = project_info['sampling_method']
        scenario_id = project_info['scenario_id']
        logger.info('*** Project info loaded ***')

        data = processed_dfs[scenario_id]
        logger.info('*** Loaded dirty dataset ***')

        # Record the user's feedback and analyze it
        s_in = data.loc[feedback.index]
        logger.info('*** Extracted sample from dataset ***')
        logger.info(feedback_dict)
        recordFeedback(data, feedback_dict, project_id,
                       current_iter, current_time)

        interpretFeedback(s_in, feedback, project_id,
                          current_iter, current_time,
                          trainer_model=trainer_model)

        # Build a new sample
        current_iter += 1
        s_out = buildSample(data, SAMPLE_SIZE,
                            project_id,
                            sampling_method=sampling_method,
                            resample=RESAMPLE)
        s_index = s_out.index

        # open file containing the indices of unserved tuples, update it and dump
        unserved_indices = pickle.load(
            open(f'{STORE_BASE_PATH}/' + project_id + '/unserved_indices.pk', 'rb'))
        unserved_indices = (set(unserved_indices) - set(s_index))
        pickle.dump(unserved_indices,
                    open(f'{STORE_BASE_PATH}/' + project_id + '/unserved_indices.pk',
                         'wb'))

        pickle.dump(s_index,
                    open(f'{STORE_BASE_PATH}/' + project_id + '/current_sample.pk', 'wb'))

        s_out.insert(0, 'id', s_out.index, True)
        logger.info(s_out.index)

        # Build feedback map for front-end
        start_time = pickle.load(
            open(f'{STORE_BASE_PATH}/' + project_id + '/start_time.pk', 'rb'))
        elapsed_time = current_time - start_time
        feedback = list()
        interaction_metadata = pickle.load(
            open(f'{STORE_BASE_PATH}/' + project_id + '/interaction_metadata.pk', 'rb'))
        interaction_metadata['user_hypothesis_history'].append(
            StudyMetric(iter_num=current_iter - 1,
                        value=[current_user_h, user_h_comment],
                        elapsed_time=elapsed_time))  # current iter - 1 because it's for the prev iter (i.e. before incrementing current_iter)
        pickle.dump(interaction_metadata,
                    open(f'{STORE_BASE_PATH}/' + project_id + '/interaction_metadata.pk',
                         'wb'))

        for idx in s_out.index:
            for col in s_out.columns:
                feedback.append({
                    'row': idx,
                    'col': col,
                    'marked': False if col == 'id' else bool(
                        interaction_metadata['feedback_recent'][idx][
                            col])})

        logger.info('*** Feedback object created ***')

        # Check if the scenario is done
        # if current_iter <= 5:
        #     terminate = False
        # else:
        #     terminate = helpers.checkForTermination(project_id)
        if current_iter > TOTAL_ITERATIONS or ((len(unserved_indices) == 0) and (not RESAMPLE)):
            msg = '[DONE]'
            val_idxs = validation_indices_dict[scenario_id]

            study_metrics = json.load(
                open(f'{STORE_BASE_PATH}/' + project_id + '/study_metrics.json', 'r'))
            

            if project_info['is_global']:
                with open(f'{STORE_BASE_PATH}/{project_id}/iteration_fd_metadata/learner/model_{current_iter-1}.pk', 'rb') as fp:
                    converged_learner_model = pickle.load(fp)

                os.makedirs("../converged_models", exist_ok=True)
                with open(f"../converged_models/converged_global_learner_{scenario_id}.json", "w") as fp:
                    json.dump(converged_learner_model, fp)
            else:
                # with open(f'../converged_models/converged_global_trainer_{scenario_id}.json', 'r') as fp:
                #     converged_global_model = json.load(fp)
                with open(f'../converged_models/converged_global_learner_{scenario_id}.json', 'r') as fp:
                    converged_global_model = json.load(fp)

                # with open(f'{STORE_BASE_PATH}/{project_id}/iteration_fd_metadata/learner/model_{current_iter-1}.pk', 'rb') as fp:
                #     converged_learner_model = pickle.load(fp)

                for i in range(current_iter):
                    with open(f'{STORE_BASE_PATH}/{project_id}/iteration_fd_metadata/learner/model_{i}.pk', 'rb') as fp:
                        learner_model = pickle.load(fp)
                    accuracy, recall, precision, f1 = \
                        compute_metrics_using_converged_model(
                            indices=val_idxs,
                            model=deepcopy(
                                learner_model),
                            converged_model=deepcopy(
                                converged_global_model),
                            scenario_id=scenario_id,
                            top_k=MODEL_FDS_TOP_K)
                    study_metrics['iter_accuracy_converged'].append(accuracy)
                    study_metrics['iter_recall_converged'].append(recall)
                    study_metrics['iter_precision_converged'].append(precision)
                    study_metrics['iter_f1_converged'].append(f1)
                json.dump(study_metrics,
                        open(f'{STORE_BASE_PATH}/' + project_id + '/study_metrics.json', 'w'))

        else:
            msg = '[SUCCESS]: Saved feedback and built new sample.'

        # Save object updates
        pickle.dump(current_iter,
                    open(f'{STORE_BASE_PATH}/' + project_id + '/current_iter.pk', 'wb'))
        with open(f'{STORE_BASE_PATH}/' + project_id + '/project_info.json', 'w') as f:
            json.dump(project_info, f)

        # Return information to the user
        response = {
            'sample': s_out.to_json(orient='index'),
            'feedback': json.dumps(feedback),
            'msg': msg
        }
        return response, 200, {'Access-Control-Allow-Origin': '*'}
