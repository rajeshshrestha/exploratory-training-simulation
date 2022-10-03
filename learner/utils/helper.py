import random
import json
import math
import pickle
import random
import time
from operator import itemgetter

import numpy as np
from scipy.stats import binom
from .initialize_variables import processed_dfs
from .initialize_variables import scenarios


# Calculate the initial prior (alpha/beta) for an FD
def initialPrior(mu, variance):
    if mu == 1:
        mu = 0.9999
    elif mu == 0:
        mu = 0.0001
    beta = (1 - mu) * ((mu * (1 - mu) / variance) - 1)
    alpha = (mu * beta) / (1 - mu)
    return abs(alpha), abs(beta)


# StudyMetric: a standardized class to represent various metrics being collected in the study
class StudyMetric(object):
    def __init__(self, iter_num, value, elapsed_time):
        self.iter_num = iter_num  # iteration number
        self.value = value  # the metric value
        # time elapsed since the beginning of the interaction
        self.elapsed_time = elapsed_time

    # Convert class object to a dictionary
    def asdict(self):
        return {
            'iter_num': self.iter_num,
            'value': self.value,
            'elapsed_time': self.elapsed_time
        }


# FDMeta: An object storing all important attributes and metrics for an FD
class FDMeta(object):
    def __init__(self, fd, a, b):

        self.fd = fd
        # LHS and RHS of the FD (not in set form)
        self.lhs = fd.split(' => ')[0][1:-1].split(', ')
        self.rhs = fd.split(' => ')[1].split(', ')

        # Beta distribution parameters
        self.alpha = a
        self.alpha_history = [StudyMetric(
            iter_num=0, value=self.alpha, elapsed_time=0)]
        self.beta = b
        self.beta_history = [StudyMetric(
            iter_num=0, value=self.beta, elapsed_time=0)]
        self.conf = (a / (a + b))
        self.conf_history = [StudyMetric(
            iter_num=0, value=self.conf, elapsed_time=0)]

        self.label_accuracy_history = [StudyMetric(
            iter_num=0, value=self.conf, elapsed_time=0)]

    # Convert class object to dictionary

    def asdict(self):
        alpha_history = list()
        for a in self.alpha_history:
            alpha_history.append(a.asdict())

        beta_history = list()
        for b in self.beta_history:
            beta_history.append(b.asdict())

        conf_history = list()
        for c in self.conf_history:
            conf_history.append(c.asdict())

        label_accuracy_history = list()
        for accuracy in self.label_accuracy_history:
            label_accuracy_history.append(accuracy.asdict())

        return {
            'fd': self.fd,
            'lhs': self.lhs,
            'rhs': self.rhs,
            'alpha': self.alpha,
            'alpha_history': alpha_history,
            'beta': self.beta,
            'beta_history': beta_history,
            'conf': self.conf,
            'conf_history': conf_history,
            'label_accuracy_history': label_accuracy_history,
        }


# CellFeedback: An instance of feedback for a particular cell
class CellFeedback(object):
    def __init__(self, iter_num, marked, elapsed_time):
        self.iter_num = iter_num  # iteration number
        # whether or not the user marked the cell as noisy in this iteration
        self.marked = marked
        # how much time has elapsed since the beginning of the interaction
        self.elapsed_time = elapsed_time

    # Convert class object to a dictionary
    def asdict(self):
        return {
            'iter_num': self.iter_num,
            'marked': self.marked,
            'elapsed_time': self.elapsed_time
        }


# Record user feedback
def recordFeedback(data, feedback, project_id, current_iter,
                   current_time):
    interaction_metadata = pickle.load(
        open('./store/' + project_id + '/interaction_metadata.p', 'rb'))
    # study_metrics = pickle.load( open('./store/' + project_id + '/study_metrics.p', 'rb') )
    start_time = pickle.load(
        open('./store/' + project_id + '/start_time.p', 'rb'))

    # Calculate elapsed time
    elapsed_time = current_time - start_time

    # Store user feedback
    # for idx in data.index:
    #     if idx in feedback.keys():
    #         for col in data.columns:
    #             interaction_metadata['feedback_history'][idx][col].append(
    #                 CellFeedback(
    #                     iter_num=current_iter,
    #                     marked=bool(feedback[idx][col]),
    #                     elapsed_time=elapsed_time))
    #     else:
    #         for col in data.columns:
    #             interaction_metadata['feedback_history'][idx][col].append(
    #                 CellFeedback(
    #                     iter_num=current_iter, marked=interaction_metadata['feedback_history'][idx][col][
    #                         -1].marked if current_iter > 1 else False,
    #                     elapsed_time=elapsed_time))

    for idx in feedback.keys():
        for col in data.columns:
            interaction_metadata['feedback_history'][idx][col].append(
                CellFeedback(
                    iter_num=current_iter,
                    marked=bool(feedback[idx][col]),
                    elapsed_time=elapsed_time))
            interaction_metadata['feedback_recent'][idx][col] = bool(feedback[idx][col])


    # Store latest sample in sample history
    interaction_metadata['sample_history'].append(
        StudyMetric(iter_num=current_iter, value=[
            int(idx) for idx in feedback.keys()], elapsed_time=elapsed_time))
    print('*** Latest feedback saved ***')

    pickle.dump(interaction_metadata, open(
        './store/' + project_id + '/interaction_metadata.p', 'wb'))
    print('*** Interaction metadata updates saved ***')


# Interpret user feedback and update alphas and betas for each FD in the hypothesis space


def interpretFeedback(s_in, feedback, project_id, current_iter,
                      current_time):
    fd_metadata = pickle.load(
        open('./store/' + project_id + '/fd_metadata.p', 'rb'))
    start_time = pickle.load(
        open('./store/' + project_id + '/start_time.p', 'rb'))
    with open('./store/' + project_id + '/project_info.json', 'r') as f:
        project_info = json.load(f)
        scenario_id = project_info['scenario_id']

    elapsed_time = current_time - start_time

    # Remove marked cells from consideration
    print('*** about to interpret feedback ***')
    marked_rows = set()
    for idx in feedback.index:
        for col in feedback.columns:
            if bool(feedback.at[idx, col]) is True:
                marked_rows.add(idx)
                break

    # Calculate P(X | \theta_h) for each FD
    for fd, fd_m in fd_metadata.items():
        successes = 0  # number of tuples that are not in a violation of this FD in the sample
        failures = 0  # number of tuples that ARE in a violation of this FD in the sample

        # Calculate which pairs have been marked and remove them from calculation
        removed_pairs = set()

        for x in marked_rows:
            for y in s_in.index:
                x_, y_ = (x, y) if x < y else (y, x)
                if (x_, y_) in scenarios[scenario_id]['hypothesis_space'][fd]['violation_pairs']:
                    removed_pairs.add((x_, y_))

        # Calculate successes and failures (to use for updating alpha and beta)
        for i in s_in.index:
            if i in marked_rows:
                continue
            # Todo: Adjust this logic of defining success
            if (len(scenarios[scenario_id]['hypothesis_space'][fd]['supports'].get(i, []))-len(scenarios[scenario_id]['hypothesis_space'][fd]['violations'].get(i, []))) >= 0:  # tuple is clean
                successes += 1
            else:
                # tuple is dirty but it's part of a vio that the user caught (i.e. they marked the wrong tuple as the error but still found the vio)
                if len([x for x in removed_pairs if i in x]) > 0:
                    successes += 1
                else:  # tuple is dirty and they missed the vio, or the vio isn't in a pair in the sample
                    failures += 1

        # Update alpha and beta
        fd_m.alpha += successes
        fd_m.alpha_history.append(StudyMetric(
            iter_num=current_iter, value=fd_m.alpha,
            elapsed_time=elapsed_time))
        fd_m.beta += failures
        fd_m.beta_history.append(StudyMetric(
            iter_num=current_iter, value=fd_m.beta, elapsed_time=elapsed_time))
        fd_m.conf = fd_m.alpha / (fd_m.alpha + fd_m.beta)
        fd_m.conf_history.append(StudyMetric(
            iter_num=current_iter, value=fd_m.conf, elapsed_time=elapsed_time))
        print(
            f'FD: {fd}, conf: {fd_m.conf}, alpha: {fd_m.alpha}, beta: {fd_m.beta}')

    # Save updated alpha/beta metrics
    pickle.dump(fd_metadata, open(
        './store/' + project_id + '/fd_metadata.p', 'wb'))


# Build a sample

def buildSample(data, sample_size, project_id,
                sampling_method,
                resample):
    if sampling_method == 'ACTIVELR':
        s_index = returnActiveLearningTuples(
            sample_size, project_id, resample=resample)
    elif sampling_method == 'RANDOM':
        s_index = returnRandomSamples(
            sample_size, project_id, resample=resample)
    else:
        raise Exception(
            f"Unknown sampling method passed: {sampling_method}!!!")

    s_out = data.loc[s_index, :]

    print('IDs of tuples in next sample:', s_out.index)

    return s_out


# Return Random sampling of tuples:
def returnRandomSamples(sample_size, project_id, resample=False):
    if resample:
        with open('./store/' + project_id + '/project_info.json', 'r') as f:
            project_info = json.load(f)
            scenario_id = project_info['scenario_id']

        '''Read current fd metadata of the project'''
        data_indices = processed_dfs[scenario_id].indices
        s_out = random.sample(list(data_indices), min(
            len(data_indices), sample_size))

    else:
        unserved_indices = pickle.load(
            open('./store/' + project_id + '/unserved_indices.p', 'rb'))
        s_out = random.sample(list(unserved_indices), min(
            len(unserved_indices), sample_size))

    return list(s_out)


# Return tuples in using active learning
def returnActiveLearningTuples(sample_size, project_id, model='SimpleAgg',
                               resample=False):
    '''Read current fd metadata of the project'''
    fd_metadata = pickle.load(
        open('./store/' + project_id + '/fd_metadata.p', 'rb'))

    unserved_indices = pickle.load(
        open('./store/' + project_id + '/unserved_indices.p', 'rb'))

    '''Compute the active learning '''
    if model == 'Bayesian':
        overall_entropy_dict = dict()
        overall_violation_pairs = set()
        for fd, metadata_obj in fd_metadata.items():
            '''Find all the violation pairs of all the hypothesis in hypothesis space'''
            overall_violation_pairs |= set(metadata_obj.violation_pairs)

            entropy_dict = compute_active_learning_indices(metadata_obj)

            '''Add to the overall_entropy_dict weighing using confidence of fd itself'''
            alpha, beta = metadata_obj.alpha, metadata_obj.beta
            for idx, entropy_value in entropy_dict.items():
                '''Skip index if resample is disabled and index already presented to the user'''
                if (not resample) and (idx not in unserved_indices):
                    continue
                if idx not in overall_entropy_dict:
                    overall_entropy_dict[idx] = (alpha) / (
                        alpha + beta) * entropy_value
                else:
                    overall_entropy_dict[idx] += (alpha) / \
                                                 (alpha + beta) * entropy_value
    elif model == 'FP':
        max_conf, max_fd_obj = -1, None
        for fd, metadata_obj in fd_metadata.items():
            if metadata_obj.conf > max_conf:
                max_conf = metadata_obj.conf
                max_fd_obj = metadata_obj
        overall_entropy_dict = compute_active_learning_indices(max_fd_obj)
        overall_violation_pairs = set(max_fd_obj.violation_pairs)
    elif model == 'SimpleAgg':
        overall_entropy_dict = dict()
        overall_violation_pairs = set()
        for fd, metadata_obj in fd_metadata.items():
            '''Find all the violation pairs of all the hypothesis in hypothesis space'''
            overall_violation_pairs |= set(metadata_obj.violation_pairs)

            entropy_dict = compute_active_learning_indices(metadata_obj)

            '''Add to the overall_entropy_dict weighing using confidence of fd itself'''
            alpha, beta = metadata_obj.alpha, metadata_obj.beta
            for idx, entropy_value in entropy_dict.items():
                '''Skip index if resample is disabled and index already presented to the user'''
                if (not resample) and (idx not in unserved_indices):
                    continue
                if idx not in overall_entropy_dict:
                    overall_entropy_dict[idx] = entropy_value
                else:
                    overall_entropy_dict[idx] += entropy_value

    '''Get top sample_size indices'''
    sorted_entropy_items = sorted(
        overall_entropy_dict.items(), key=itemgetter(1), reverse=True)

    '''Sampled data from the violation pairs that has top n entropy values'''
    s_out = set()
    for idx, entropy_value in sorted_entropy_items:
        '''Sample from the violation pairs that has the idx in it'''
        candidate_violation_pairs = [
            vio_pair for vio_pair in overall_violation_pairs if
            idx in vio_pair]
        if len(candidate_violation_pairs) == 0:
            continue
        new_data = random.sample(candidate_violation_pairs, 1)[0]

        if not resample:
            if (new_data[0] not in unserved_indices) or (
                    new_data[1] not in unserved_indices):
                continue
        s_out |= set(new_data)

        if len(s_out) >= sample_size:
            break

    return list(s_out)


def compute_active_learning_indices(single_fd_metadata, n=None):
    '''Compute the current probability of fd being the correct one'''
    alpha = single_fd_metadata.alpha
    beta = single_fd_metadata.beta
    p = alpha / (alpha + beta)

    '''Entropy dict for active learning'''
    entropy_dict = dict()

    '''Iterate through all the tuples'''
    for idx in single_fd_metadata.data_indices:
        total_compliance_num = len(single_fd_metadata.supports.get(idx))
        total_violation_num = len(single_fd_metadata.violations.get(idx))

        '''Compute probability of tuple being clean based on compliance on other tuples and the probability of fd being the correct one'''
        clean_prob = binom.pmf(k=total_compliance_num, n=(
            total_compliance_num + total_violation_num), p=p)

        '''Compute entropy'''
        entropy = - clean_prob * \
            np.log2(clean_prob) - (1 - clean_prob) * np.log2(
                1 - clean_prob)
        entropy_dict[idx] = entropy

    if n is not None:
        '''Find top values'''
        top_n_entropy_dict = dict(
            sorted(entropy_dict.items(), key=itemgetter(1), reverse=True)[:n])
        return top_n_entropy_dict
    else:
        return entropy_dict
