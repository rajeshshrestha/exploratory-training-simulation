import json
import random
import re
import sys

import requests
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import os

from initialize_variables import scenarios, models_dict

# Build the feedback dictionary object that will be utilized during the
# interaction
def buildFeedbackMap(data, feedback, header):
    feedbackMap = dict()
    cols = header
    for row in data.keys():
        tup = dict()
        for col in cols:
            trimmedCol = re.sub(r'/[\n\r]+/g', '', col)
            cell = next(f for f in feedback if
                        f['row'] == data[row]['id'] and f[
                            'col'] == trimmedCol)
            tup[col] = cell['marked']
        feedbackMap[row] = tup
    return feedbackMap


def run(s, b_type):
    if s is None:
        s = 'omdb'

    iter_num = 0

    # Start the interaction

    '''
        Initialize the project and the user intital prior belief
    '''
    try:
        r = requests.post('http://localhost:5000/duo/api/import', data={
            'scenario_id': s,
            'email': '',
            'initial_fd': 'Not Sure',
            # TODO: Logic for what the simulated user thinks at first
            'fd_comment': '',
            'skip_user': True,
            # Skip user email handling since this is not part of the study
            'violation_ratio': 'close'
        })
        res = r.json()
        project_id = res['project_id']
    except Exception as e:
        print(e)
        return

    feedback = None

    # Get first sample
    try:
        r = requests.post('http://localhost:5000/duo/api/sample',
                          data={'project_id': project_id})
        res = r.json()

        sample = res['sample']
        data = json.loads(sample)
        feedback = json.loads(res['feedback'])

        # ! Change the sample to string type with None replaced by '' for
        # all the fields in the sample except for the id field
        for row in data.keys():
            for j in data[row].keys():
                if data[row][j] is None:
                    data[row][j] = ''
                elif type(data[row][j]) != 'str':
                    data[row][j] = str(data[row][j])

        print('prepped data')
    except Exception as e:
        print(e)
        return

    print('initialized feedback object')
    msg = ''
    iter_num = 0

    # Get table column names
    header = list()
    for row in data.keys():
        header = [c for c in data[row].keys() if c != 'id']
        break

    # Begin iterations and continue till done
    while msg != '[DONE]' and (len(data) > 0):
        iter_num += 1

        # Initialize feedback dictionary utilized during interaction
        feedbackMap = buildFeedbackMap(data, feedback,
                                       header)

        # Decide for each row whether to mark or not
        for row in data.keys():
            '''Full oracle'''
            if b_type == 'full-oracle':
                if not models_dict[s]["predictions"][row]:
                    for rh in header:
                        feedbackMap[row][rh] = True

        # Set up the feedback representation that will be given to the
        # server
        feedback = dict()
        for f in feedbackMap.keys():
            feedback[data[f]['id']] = feedbackMap[f]

        formData = {
            'project_id': project_id,
            'feedback': json.dumps(feedback),
            'current_user_h': 'Not Sure',
            # TODO: Hypothesize an FD in the simulation in each iteration
            'user_h_comment': '',
        }

        try:
            r = requests.post('http://localhost:5000/duo/api/feedback',
                              data=formData)  # Send feedback to server
            res = r.json()
            msg = res['msg']
            if msg != '[DONE]':  # Server says do another iteration
                sample = res['sample']
                feedback = json.loads(res['feedback'])
                data = json.loads(sample)

                for row in data.keys():
                    for j in data[row].keys():
                        if j == 'id':
                            continue

                        if data[row][j] is None:
                            data[row][j] = ''
                        elif type(data[row][j]) != 'str':
                            data[row][j] = str(data[row][j])

        except Exception as e:
            print(e)
            msg = '[DONE]'


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

    s = sys.argv[1]  # Scenario #
    b_type = sys.argv[
        2]  # Bayesian type ("oracle", "informed", "uninformed", "random")
    decision_type = sys.argv[3]  # Decision type ("coin-flip" or "threshold")
    num_runs = int(sys.argv[4])  # How many runs of this simulation to do
    stat_calc = None if len(sys.argv) < 6 else sys.argv[
        5]  # Are we evaluating precision or recall?

    
    cpu_num = os.cpu_count()

    with Pool(10) as p:
        p.map(partial(run, b_type=b_type), [s for i in range(num_runs)])
