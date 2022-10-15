import requests
import json
import logging

logger = logging.getLogger(__file__)


def initialize_learner(scenario_id,
                       sampling_method,
                       trainer_type,
                       use_val_data,
                       trainer_prior_type,
                       learner_prior_type):
    """
    Initialize the project and the user intital prior belief
    """
    try:
        response = requests.post(
            "http://localhost:5000/duo/api/import",
            data={
                "scenario_id": scenario_id,
                "email": "",
                "initial_fd": "Not Sure",
                "fd_comment": "",
                "skip_user": True,
                "violation_ratio": "close",
                "sampling_method": sampling_method,
                "trainer_type": trainer_type,
                'use_val_data': use_val_data,
                "trainer_prior_type": trainer_prior_type,
                "learner_prior_type": learner_prior_type
            },
        ).json()
        project_id = response["project_id"]
        return project_id, response
    except Exception as e:
        logger.error(e)
        raise Exception(e)


def get_initial_sample(project_id):
    # Get first sample
    try:
        response = requests.post('http://localhost:5000/duo/api/sample',
                                 data={'project_id': project_id}).json()

        sample = response['sample']
        data = json.loads(sample)
        feedback = json.loads(response['feedback'])

        # ! Change the sample to string type with None replaced by '' for
        # all the fields in the sample except for the id field
        for row in data.keys():
            for j in data[row].keys():
                if data[row][j] is None:
                    data[row][j] = ''
                elif type(data[row][j]) != 'str':
                    data[row][j] = str(data[row][j])

        logger.info('prepped data')

        '''Find the column names excluding the id in the data'''
        for row in data.keys():
            header = [c for c in data[row].keys() if c != 'id']
            break

        return data, header, feedback
    except Exception as e:
        logger.info(e)
        raise Exception(e)


def send_feedback(project_id, feedback_dict, model_dict):

    formData = {
        'project_id': project_id,
        'feedback': json.dumps(feedback_dict),
        'current_user_h': 'Not Sure',
        # TODO: Hypothesize an FD in the simulation in each iteration
        'user_h_comment': '',
        'trainer_model': json.dumps(model_dict)
    }

    try:
        response = requests.post('http://localhost:5000/duo/api/feedback',
                                 data=formData).json()
        msg = response['msg']
        if msg != '[DONE]':  # Server says do another iteration
            sample = response['sample']
            feedback = json.loads(response['feedback'])
            data = json.loads(sample)

            for row in data.keys():
                for j in data[row].keys():
                    if j == 'id':
                        continue

                    if data[row][j] is None:
                        data[row][j] = ''
                    elif type(data[row][j]) != 'str':
                        data[row][j] = str(data[row][j])
            return data, feedback, msg
        else:
            return None, None, msg

    except Exception as e:
        logger.error(e)
        msg = '[DONE]'
        return None, None, msg
