import json
import pickle
import random
import re
import sys

import numpy as np
import requests
from multiprocessing import Process
from tqdm import tqdm


# FD Metadata object
class FDMetaUser(object):
    def __init__(self, fd, a, b, support, vios, vio_pairs):
        self.lhs = fd.split(' => ')[0][1:-1].split(', ')
        self.rhs = fd.split(' => ')[1].split(', ')
        self.alpha = a
        self.alpha_history = [a]
        self.beta = b
        self.beta_history = [b]
        self.conf = (a / (a + b))
        self.support = support
        self.vios = vios
        self.vio_pairs = vio_pairs

    # Convert class object to dictionary
    def asdict(self):
        return {
            'lhs': self.lhs,
            'rhs': self.rhs,
            'alpha': self.alpha,
            'alpha_history': self.alpha_history,
            'beta': self.beta,
            'beta_history': self.beta_history,
            'conf': self.conf,
            'support': self.support,
            'vios': self.vios,
            'vio_pairs': [list(vp) for vp in self.vio_pairs]
        }


# Calculate the initial probability mean for the FD
def initialPrior(mu, variance):
    beta = (1 - mu) * ((mu * (1 - mu) / variance) - 1)
    alpha = (mu * beta) / (1 - mu)
    return alpha, beta


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
                        f['row'] == int(data[row]['id']) and f[
                            'col'] == trimmedCol)
            tup[col] = cell['marked']
        feedbackMap[row] = tup
    return feedbackMap


def shuffleFDs(fds):
    return random.shuffle(fds)


# - s: scenario #
# - b_type: Bayesian type (i.e. what kind of user are we simulating)
#           ("oracle", "informed", "uninformed")
# - decision_type: coin-flip or probability threshold decision making
#                   ("coin-flip", "threshold")
# - stat_calc: are we evaluating precision or recall?
def run(s, b_type, decision_type, stat_calc):
    '''
        Set p_max value based on the Bayesian type:
            orcale: Oracle user always right
                p_max = 0.9
            informed: Informed user having some background knowledge
                p_max = 0.9
            else:
                p_max = 0.5
    '''
    if b_type == 'oracle':  # Oracle user; always right
        p_max = 0.9
    elif b_type == 'informed':  # Informed user; assuming has some
                                # background knowledge
        p_max = 0.9
    else:  # Uninformed user; assuming no prior knowledge
        p_max = 0.5

    '''
        Set Scene to 0 if not set
        Read the scenario data with target_fd and hypothesis_space
    '''
    if s is None:
        s = '0'
    with open('../scenarios.json', 'r') as f:
        scenarios = json.load(f)
    scenario = scenarios[s]
    target_fd = scenario['target_fd']  # ! target fd
    h_space = scenario[
        'hypothesis_space']  # ! fd space with their info in dirty data

    fd_metadata = dict()

    iter_num = 0

    # Get initial probabilities for FDs in hypothesis space and build their
    # metadata objects

    '''
        Obtain the target_fd info from the hypothesis space
        #? Maybe simulating for target_fd or not only
    '''
    for h in h_space:
        if h['cfd'] == target_fd:
            target_fd_m = h

        h['vio_pairs'] = set(tuple(vp) for vp in h[
            'vio_pairs'])  # Get violation pairs in the form of the sets of
        # tuples

        '''
            Compute alpha and beta for user simulation using mu and variance,
                - Get the confidence as the mu for the oracle from the clean 
                    dataset itself with variance=0.00000001
                - For informed, use the conf computed from the dirty data with 
                    variance=0.01
            else, use alpha=1 and beta=1
        '''
        if b_type == 'oracle':
            mu = next(i for i in scenario['clean_hypothesis_space'] if
                      i['cfd'] == h['cfd'])['conf']
            if mu == 1:
                mu = 0.99999
            variance = 0.00000001
            alpha, beta = initialPrior(mu, variance)
        elif b_type == 'informed':
            mu = h['conf']
            variance = 0.01
            alpha, beta = initialPrior(mu, variance)
        else:
            alpha = 1
            beta = 1

        fd_m = FDMetaUser(
            fd=h['cfd'],
            a=alpha,
            b=beta,
            support=h['support'],
            vios=h['vios'],
            vio_pairs=h['vio_pairs']
        )
        # print('iter:', iter_num)
        # print('alpha:', fd_m.alpha)
        # print('beta:', fd_m.beta)
        fd_metadata[h['cfd']] = fd_m

    project_id = None

    # Start the interaction

    '''
        Initialize the project and the user intital prior belief
    '''
    try:
        r = requests.post('http://localhost:5000/duo/api/import', data={
            'scenario_id': str(s),
            'email': '',
            'initial_fd': 'Not Sure',
            # TODO: Logic for what the simulated user thinks at first
            'fd_comment': '',
            'skip_user': True,
            # Skip user email handling since this is not part of the study
            'violation_ratio': 'close'
            # TODO: Add command-line argument for close or far violation
            #  ratio (e.g. 3:1 or 3:2). This may change.
        })
        res = r.json()
        project_id = res['project_id']
    except Exception as e:
        print(e)
        return


    data = None
    feedback = None
    sample_X = list()

    # Get first sample
    try:
        '''
            Get sample, true violation pairs in that samples(in dirty data) from the api
            Also the placeholder for each cell feedback is obtained
            Sample:
                data: {
                        '2': {'facilityname': 'AKIACHAK',
                              'id': 2,
                              'manager': 'LAWRENCE DAVIS',
                              'owner': 'ALASKA DOT&PF CENTRAL REGION',
                              'type': 'AIRPORT'},
                        '22': {'facilityname': 'ANVIK',
                                'id': 22,
                                'manager': 'ERIK WEINGARTH',
                                'owner': 'ALASKA DOT&PF NORTHERN REGION',
                                'type': 'AIRPORT'},
                                ....
                    }

                sample_X = {(5, 25)}

        '''
        r = requests.post('http://localhost:5000/duo/api/sample',
                          data={'project_id': project_id})
        res = r.json()

        sample = res['sample']
        sample_X = set(tuple(x) for x in res[
            'X'])  # ## list of true violations(with correct assumed based
        # on the clean data) with respect to target FD in dirty data
        data = json.loads(sample)
        feedback = json.loads(res['feedback'])

        # ! Change the sample to string type with None replaced by '' for
        # all the fields in the sample except for the id field
        for row in data.keys():
            for j in data[row].keys():
                if j == 'id':
                    continue
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

    max_marked = 1
    mark_prob = 0.5

    marked_rows = set()
    vios_marked = set()
    vios_found = set()

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

        # Bayesian behavior
        # ? Maybe only evaluating for the target_fd like before
        max_conf, max_fd = -1, None
        old_max_conf, old_max_fd = -1, None
        for fd, fd_m in fd_metadata.items():

            # Find old best fd before updating model
            if fd_m.alpha / (fd_m.alpha + fd_m.beta) > old_max_conf:
                old_max_conf = fd_m.alpha / (fd_m.alpha + fd_m.beta)
                old_max_fd = fd

            # Step 1: update hyperparameters ? Not updating incase of
            # oracle. Does it mean oracle knows all and no need for
            # hyperparameter update.
            if b_type != 'oracle':
                successes = 0
                failures = 0

                for i in data.keys():
                    if int(i) not in fd_m.vios:
                        successes += 1
                    else:
                        failures += 1

                # print('successes:', successes)
                # print('failures:', failures)

                fd_m.alpha += successes
                fd_m.alpha_history.append(fd_m.alpha)
                fd_m.beta += failures
                fd_m.beta_history.append(fd_m.beta)
                # print('alpha:', fd_m.alpha)
                # print('beta:', fd_m.beta)

            if fd_m.alpha / (fd_m.alpha + fd_m.beta) > max_conf:
                max_conf = fd_m.alpha / (fd_m.alpha + fd_m.beta)
                max_fd = fd

        # Step 2: mark errors according to new beliefs
        selected_fd = random.choices([old_max_fd, max_fd], weights=[0.01, 0.99])[
            0]
        fd_m = fd_metadata[selected_fd]


        # ! account for update of alpha and beta incase of b_type not being
        # oracle
        # ? probability of fd i.e. target FD being true in this case (mu)
        q_t = fd_m.alpha / (
                fd_m.alpha + fd_m.beta) if b_type != 'oracle' else fd_m.conf

        iter_marked_rows = {i for i in marked_rows if str(i) in data.keys()}

        # ? Didn't understand the logic behind them
        iter_vios_marked = {(x, y) for (x, y) in vios_marked if
                            (x != y and str(x) in data.keys() and str(y) in data.keys() and (x, y) in fd_m.vio_pairs) or (
                                    x == y and str(x) in data.keys())}
        iter_vios_found = {(x, y) for (x, y) in vios_found if
                           (x != y and (x, y) in fd_m.vios_pairs and str(x) in data.keys() and str(y) in data.keys()) or (
                                   x == y and str(x) in data.keys())}
        iter_vios_total = {(x,y) for (x,y) in fd_m.vio_pairs if x in data.keys() and y in data.keys()}

        # print('q_t:', q_t)

        # Decide for each row whether to mark or not
        for row in data.keys():
            '''
            if b_type is oracle
                stat_calc=precision use q_t=mark_prob=0.5 
                else stat_calc=recall and len(iter_vios_marked) >= max_marked,
                    q_t=0
            '''

            if b_type == 'oracle':
                if stat_calc == 'precision':
                    q_t = mark_prob
                elif stat_calc == 'recall':
                    if len(iter_vios_marked) >= max_marked:
                        q_t = 0
                else:
                    continue

            vios_w_i = {v for v in iter_vios_total if
                        int(row) in v and v not in iter_vios_marked}  # Find
            # all violations that involve this row

            if decision_type == 'coin-flip':
                decision = np.random.binomial(1, q_t)
            else:
                decision = 1 if q_t >= p_max else 0

            
            '''Full oracle'''
            if b_type == 'full-oracle':
                if int(row) in target_fd_m['vios']:
                    for rh in fd_m.rhs:
                        feedbackMap[row][rh] = True
                continue



            if decision == 1:  # User correctly figured out the truth for
                # this tuple with respect to this FD
                if len(vios_w_i) > 0:  # This tuple is involved in a
                    # violation of our hypothesized FD
                    '''If decision 1 obtained and actual not marked 
                    violations present including that row is greater than 0, 
                    then mark rhs of those rows to be True '''
                    for rh in fd_m.rhs:
                        feedbackMap[row][rh] = True

                    vios_found |= vios_w_i
                    iter_vios_found |= vios_w_i
                    vios_marked |= vios_w_i
                    iter_vios_marked |= vios_w_i
                    marked_rows.add(int(row))
                    iter_marked_rows.add(int(row))
                    vios_marked.discard((int(row), int(row)))
                    iter_vios_marked.discard((int(row), int(row)))

                else:  # This tuple has no violations of our hypothesized FD
                    for rh in fd_m.rhs:
                        feedbackMap[row][rh] = False
                    vios_marked.discard((int(row), int(row)))
                    iter_vios_marked.discard((int(row), int(row)))
                    marked_rows.discard(int(row))
                    iter_marked_rows.discard(int(row))

            else:  # User did not figure it out
                if len(vios_w_i) > 0:  # This tuple is involved in a
                    # violation of our hypothesized FD
                    for rh in fd_m.rhs:
                        feedbackMap[row][rh] = False
                    vios_found -= vios_w_i
                    iter_vios_found -= vios_w_i
                    vios_marked -= vios_w_i
                    iter_vios_marked -= vios_w_i
                    marked_rows.discard(int(row))
                    iter_marked_rows.discard(int(row))

                else:  # This tuple has no violations of our hypothesized FD
                    for rh in fd_m.rhs:
                        feedbackMap[row][rh] = True
                    vios_marked.add((int(row), int(row)))
                    iter_vios_marked.add((int(row), int(row)))
                    marked_rows.add(int(row))
                    iter_marked_rows.add(int(row))

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
                sample_X = set(tuple(x) for x in res['X'])
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
                if mark_prob < 0.9:
                    mark_prob += 0.05
                elif mark_prob < 0.95:
                    mark_prob += 0.01

                if max_marked < 5:
                    max_marked += 1

        except Exception as e:
            print(e)
            msg = '[DONE]'

    # pickle.dump(fd_metadata,
    #             open('./store/' + project_id + '/fd_metadata_user.p', 'wb'))


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

    # parallel_process_number = 5
    # iteration_number = num_runs//parallel_process_number if num_runs%parallel_process_number ==0 else (num_runs//parallel_process_number) +1
    
    # for i in range(0, iteration_number):
    #     processes = [Process(target=run, args=(s, b_type, decision_type, stat_calc))]
    #     for p in processes:
    #         p.start()
    #     for p in processes:
    #         p.join()

    for i in tqdm(list(range(num_runs))):
        run(s, b_type, decision_type, stat_calc)
