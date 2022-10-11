from asyncio.tasks import _T1
import random
import json
import math
import pickle
import random
import time
from operator import itemgetter
import logging

import numpy as np
from scipy.stats import binom
from .initialize_variables import processed_dfs
from .initialize_variables import scenarios
from .initialize_variables import models_dict
from .env_variables import MODEL_FDS_TOP_K
from .env_variables import ACCURACY_ESTIMATION_SAMPLE_NUM
from .env_variables import ACTIVE_LEARNING_CANDIDATE_INDICES_NUM

# Calculate the initial prior (alpha/beta) for an FD

logger = logging.getLogger(__file__)


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
            'conf_history': conf_history
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
    # study_metrics = json.load( open('./store/' + project_id + '/study_metrics.json', 'r') )
    start_time = pickle.load(
        open('./store/' + project_id + '/start_time.p', 'rb'))

    # Calculate elapsed time
    elapsed_time = current_time - start_time

    for idx in feedback.keys():
        for col in data.columns:
            interaction_metadata['feedback_history'][idx][col].append(
                CellFeedback(
                    iter_num=current_iter,
                    marked=bool(feedback[idx][col]),
                    elapsed_time=elapsed_time))
            interaction_metadata['feedback_recent'][idx][col] = bool(
                feedback[idx][col])

    # Store latest sample in sample history
    interaction_metadata['sample_history'].append(
        StudyMetric(iter_num=current_iter, value=[
            int(idx) for idx in feedback.keys()], elapsed_time=elapsed_time))
    logger.info('*** Latest feedback saved ***')

    pickle.dump(interaction_metadata, open(
        './store/' + project_id + '/interaction_metadata.p', 'wb'))
    logger.info('*** Interaction metadata updates saved ***')


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
    logger.info('*** about to interpret feedback ***')
    marked_rows = set()
    for idx in feedback.index:
        for col in feedback.columns:
            if bool(feedback.at[idx, col]) is True:
                marked_rows.add(idx)
                break

    # Calculate P(X | \theta_h) for each FD
    mae_model_error = 0
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
        logger.info(f'FD: {fd}, conf: {fd_m.conf}, '
                    f'alpha: {fd_m.alpha}, beta: {fd_m.beta}')
        # print(f'FD: {fd}, conf: {fd_m.conf}, '
        #       f"true_conf: {models_dict['omdb']['model'][fd]}")
        mae_model_error += abs(fd_m.conf - models_dict['omdb']['model'][fd])

    logger.info(f"MAE Model Conf Error: {mae_model_error}")

    '''Compute accuracy in Unserved dataset'''
    unserved_indices = pickle.load(
        open('./store/' + project_id + '/unserved_indices.p', 'rb'))  # whether to compute on all remaining unserved indices or sample from the remaining one
    model_dict = dict((fd, fd_m.conf)for fd, fd_m in fd_metadata.items())
    validation_indices = np.random.choice(list(unserved_indices),
                                          size=min(len(unserved_indices),
                                          ACCURACY_ESTIMATION_SAMPLE_NUM),
                                          replace=False)
    accuracy, recall, precision, f1 = compute_metrics(
        indices=validation_indices,
        model=model_dict,
        scenario_id=scenario_id,
        top_k=MODEL_FDS_TOP_K)
    logger.info(
        "=============================================================================")
    logger.info(
        f"Accuracy in {len(validation_indices)} unserved data: {round(accuracy,2)}")
    logger.info(
        "=============================================================================")
    study_metrics = json.load(
        open('./store/' + project_id + '/study_metrics.json', 'rb'))
    study_metrics['iter_accuracy'].append(accuracy)
    study_metrics['iter_recall'].append(recall)
    study_metrics['iter_precision'].append(precision)
    study_metrics['iter_f1'].append(f1)
    study_metrics['iter_mae_model_error'].append(mae_model_error)
    study_metrics['elapsed_time'].append(elapsed_time)
    json.dump(study_metrics,
              open('./store/' + project_id + '/study_metrics.json', 'w'))

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

    logger.info(f'IDs of tuples in next sample: {s_out.index}')

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
def returnActiveLearningTuples(sample_size, project_id,
                               resample=False):
    '''Read current fd metadata of the project'''
    fd_metadata = pickle.load(
        open('./store/' + project_id + '/fd_metadata.p', 'rb'))

    unserved_indices = pickle.load(
        open('./store/' + project_id + '/unserved_indices.p', 'rb'))

    '''Subsample candiate unserved indices'''
    if ACTIVE_LEARNING_CANDIDATE_INDICES_NUM > 0:
        candidate_unserved_indices = set(np.random.choice(list(
            unserved_indices), size=ACTIVE_LEARNING_CANDIDATE_INDICES_NUM, replace=False))
    else:
        candidate_unserved_indices = unserved_indices

    with open('./store/' + project_id + '/project_info.json', 'r') as f:
        project_info = json.load(f)
        scenario_id = project_info['scenario_id']

    model_dict = dict((fd, fd_m.conf)for fd, fd_m in fd_metadata.items())
    top_k_model_dict = dict(
        sorted(model_dict.items(), key=itemgetter(1), reverse=True)[:MODEL_FDS_TOP_K])

    '''Compute the active learning '''
    overall_entropy_dict = compute_entropy_values(
        indices=candidate_unserved_indices, top_model_fds=top_k_model_dict, scenario_id=scenario_id)

    overall_violation_pairs = set()

    if resample:
        for fd in top_k_model_dict:
            overall_violation_pairs |= scenarios[scenario_id]['hypothesis_space'][fd]['violation_pairs']
    else:
        for fd in top_k_model_dict:
            overall_violation_pairs |= set([pair for pair in scenarios[scenario_id]['hypothesis_space'][fd]['violation_pairs'] if pair[0]
                                           in unserved_indices and pair[1] in unserved_indices and (pair[0] in candidate_unserved_indices or pair[1] in candidate_unserved_indices)])

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

        s_out |= set(new_data)

        if len(s_out) >= sample_size:
            break

    '''If insufficient sample remaining from top k entropy indices'''
    remaining = sample_size-len(s_out)
    if remaining > 0:
        missing_ones = [idx for idx, entropy_value in sorted_entropy_items
                        if idx not in s_out]
        s_out |= set(missing_ones[:remaining])

    return list(s_out)


def compute_conditional_clean_prob(idx, fd, fd_prob, scenario_id, data_indices=None):
    if data_indices is None:
        compliance_num = len(
            scenarios[scenario_id]['hypothesis_space'][fd]['supports'].get(idx, []))
        violation_num = len(
            scenarios[scenario_id]['hypothesis_space'][fd]['violations'].get(idx, []))
    else:
        compliance_num = len([idx_ for idx_ in scenarios[scenario_id]['hypothesis_space']
                             [fd]['supports'].get(idx, []) if idx_ in data_indices])
        violation_num = len([idx_ for idx_ in scenarios[scenario_id]['hypothesis_space']
                            [fd]['violations'].get(idx, []) if idx_ in data_indices])

    tuple_clean_score = math.exp(fd_prob*(compliance_num-violation_num))
    tuple_dirty_score = math.exp(fd_prob*(-compliance_num+violation_num))
    cond_p_clean = tuple_clean_score/(tuple_clean_score+tuple_dirty_score)

    return cond_p_clean


def get_average_cond_clean_prediction(indices, model, scenario_id):
    conditional_clean_probability_dict = dict()
    indices = set(indices)
    for idx in indices:
        cond_clean_prob = np.mean([compute_conditional_clean_prob(
            idx=idx, fd=fd, fd_prob=fd_prob, scenario_id=scenario_id,
            data_indices=indices) for fd, fd_prob in model.items()])  # whether to include the validation_indices or all the data_indices while computing the conditional clean probability
        conditional_clean_probability_dict[idx] = cond_clean_prob
    return conditional_clean_probability_dict


def compute_metrics(indices, model, scenario_id, top_k):
    '''Pick top k fds if needed'''
    if 0 < top_k < len(model):
        model = dict(sorted(model.items(), key=itemgetter(1),
                     reverse=True)[:top_k])

    conditional_clean_probability_dict = get_average_cond_clean_prediction(
        indices=indices, model=model, scenario_id=scenario_id)
    logger.info(conditional_clean_probability_dict)

    idxs = list(conditional_clean_probability_dict.keys())
    is_dirty_true_labels = [not models_dict[scenario_id]["predictions"][idx]
                            for idx in idxs]
    is_dirty_predicted_labels = [conditional_clean_probability_dict[idx] < 0.5
                                 for idx in idxs]
    is_correct = [true_val == predicted_val for true_val, predicted_val
                  in zip(is_dirty_true_labels, is_dirty_predicted_labels)]  # ignore indices with probability=0.5 as this means most probably, there wasn't any compliance and violations in the validation data for the tuple
    true_positive = [true_val and predicted_val for true_val, predicted_val
                     in zip(is_dirty_true_labels, is_dirty_predicted_labels)]

    # print(set(is_correct))
    # print(set(true_positive))
    # print(set(is_dirty_true_labels))
    # print(set(is_dirty_predicted_labels))

    accuracy = np.mean(is_correct)
    recall = sum(true_positive)/sum(is_dirty_true_labels)
    precision = sum(true_positive)/sum(is_dirty_predicted_labels)
    f1 = 2*recall*precision/(recall+precision)

    return accuracy, recall, precision, f1


def compute_entropy_values(indices, top_model_fds, scenario_id):

    conditional_clean_probability_dict = get_average_cond_clean_prediction(
        indices=indices, model=top_model_fds, scenario_id=scenario_id)

    probabilities = np.array([conditional_clean_probability_dict[idx]
                              for idx in indices])
    entropies = -probabilities * \
        np.log2(probabilities)-(1-probabilities)*np.log2(1-probabilities)
    entropy_dict = dict((idx, entropy_val)
                        for idx, entropy_val in zip(indices, entropies))

    return entropy_dict
