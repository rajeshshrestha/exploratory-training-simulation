import json
import random
import pickle
import numpy as np
from operator import itemgetter


from .initialize_variables import processed_dfs
from .metrics import compute_entropy_values
from .metrics import get_average_cond_clean_prediction
from .initialize_variables import scenarios
from .env_variables import MODEL_FDS_TOP_K
from .env_variables import ACTIVE_LEARNING_CANDIDATE_INDICES_NUM
from .env_variables import STOCHASTIC_BEST_RESPONSE_CANDIDATE_INDICES_NUM
from .env_variables import STOCHASTIC_ACTIVE_LEARNING_CANDIDATE_INDICES_NUM
from .env_variables import STOCHASTIC_BEST_RESPONSE_GAMMA
from .env_variables import STOCHASTIC_UNCERTAINTY_SAMPLING_GAMMA

# Return Samples using stochastic best response


def returnStochasticBRTuples(sample_size, project_id, resample=False):
    '''Read current fd metadata of the project'''
    fd_metadata = pickle.load(
        open('./store/' + project_id + '/fd_metadata.pk', 'rb'))

    unserved_indices = pickle.load(
        open('./store/' + project_id + '/unserved_indices.pk', 'rb'))

    with open('./store/' + project_id + '/project_info.json', 'r') as f:
        project_info = json.load(f)
        scenario_id = project_info['scenario_id']

    if not resample:
        '''Subsample candidate unserved indices'''
        if STOCHASTIC_BEST_RESPONSE_CANDIDATE_INDICES_NUM > 0:
            candidate_unserved_indices = set(np.random.choice(list(
                unserved_indices),
                size=min(ACTIVE_LEARNING_CANDIDATE_INDICES_NUM,
                         len(unserved_indices)),
                replace=False))
        else:
            candidate_unserved_indices = unserved_indices
    else:
        candidate_unserved_indices = set(processed_dfs[scenario_id].index)

    '''Compute top k fd model dict'''
    model_dict = dict((fd, fd_m.conf)for fd, fd_m in fd_metadata.items())
    top_k_model_dict = dict(
        sorted(model_dict.items(),
               key=itemgetter(1), reverse=True)[:MODEL_FDS_TOP_K])

    '''Compute the overall violation pairs'''
    overall_violation_pairs = set()

    if resample:
        for fd in top_k_model_dict:
            overall_violation_pairs |= scenarios[scenario_id]['hypothesis_space'][fd]['violation_pairs']
    else:
        for fd in top_k_model_dict:
            overall_violation_pairs |= set([pair for pair in scenarios[scenario_id]['hypothesis_space'][fd]['violation_pairs'] if pair[0]
                                           in unserved_indices and pair[1] in unserved_indices and (pair[0] in candidate_unserved_indices or pair[1] in candidate_unserved_indices)])

    '''Get predictions of the unserved indices'''
    conditional_clean_probability_dict = get_average_cond_clean_prediction(
        indices=candidate_unserved_indices,
        model=top_k_model_dict,
        scenario_id=scenario_id)

    '''Vectorize the computation of the probability using numpy'''
    indices = list(conditional_clean_probability_dict.keys())
    dirty_prob = 1 - np.array([conditional_clean_probability_dict[idx]
                              for idx in indices])
    sample_prob = np.exp(dirty_prob/STOCHASTIC_BEST_RESPONSE_GAMMA)
    sample_prob = sample_prob/np.sum(sample_prob)

    '''Sample data using the sample probability'''
    s_out = set()
    sampled_indices = set(np.random.choice(
        indices, sample_size, replace=False, p=sample_prob))
    for idx in sampled_indices:
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
        missing_ones = [idx for idx in sampled_indices.difference(s_out)]
        s_out |= set(missing_ones[:remaining])

    return list(s_out)


# Return Random sampling of tuples:
def returnRandomTuples(sample_size, project_id, resample=False):
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
            open('./store/' + project_id + '/unserved_indices.pk', 'rb'))
        s_out = random.sample(list(unserved_indices), min(
            len(unserved_indices), sample_size))

    return list(s_out)


# Return tuples in using active learning
def returnActiveLearningTuples(sample_size, project_id,
                               resample=False):
    '''Read current fd metadata of the project'''
    fd_metadata = pickle.load(
        open('./store/' + project_id + '/fd_metadata.pk', 'rb'))

    unserved_indices = pickle.load(
        open('./store/' + project_id + '/unserved_indices.pk', 'rb'))

    '''Subsample candiate unserved indices'''
    if not resample:
        if ACTIVE_LEARNING_CANDIDATE_INDICES_NUM > 0:
            candidate_unserved_indices = set(np.random.choice(list(
                unserved_indices),
                size=min(ACTIVE_LEARNING_CANDIDATE_INDICES_NUM,
                         len(unserved_indices)),
                replace=False))
        else:
            candidate_unserved_indices = unserved_indices
    else:
        candidate_unserved_indices = set(processed_dfs[scenario_id].index)

    with open('./store/' + project_id + '/project_info.json', 'r') as f:
        project_info = json.load(f)
        scenario_id = project_info['scenario_id']

    model_dict = dict((fd, fd_m.conf)for fd, fd_m in fd_metadata.items())
    top_k_model_dict = dict(
        sorted(model_dict.items(),
               key=itemgetter(1), reverse=True)[:MODEL_FDS_TOP_K])

    '''Compute the active learning '''
    overall_entropy_dict = compute_entropy_values(
        indices=candidate_unserved_indices,
        top_model_fds=top_k_model_dict,
        scenario_id=scenario_id)

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


# Return tuples in using active learning
def returnStochasticActiveLRTuples(sample_size, project_id,
                                   resample=False):
    '''Read current fd metadata of the project'''
    fd_metadata = pickle.load(
        open('./store/' + project_id + '/fd_metadata.pk', 'rb'))

    unserved_indices = pickle.load(
        open('./store/' + project_id + '/unserved_indices.pk', 'rb'))

    '''Subsample candiate unserved indices'''
    if not resample:
        if STOCHASTIC_ACTIVE_LEARNING_CANDIDATE_INDICES_NUM > 0:
            candidate_unserved_indices = set(np.random.choice(list(
                unserved_indices),
                size=min(STOCHASTIC_ACTIVE_LEARNING_CANDIDATE_INDICES_NUM,
                         len(unserved_indices)),
                replace=False))
        else:
            candidate_unserved_indices = unserved_indices
    else:
        candidate_unserved_indices = set(processed_dfs[scenario_id].index)

    with open('./store/' + project_id + '/project_info.json', 'r') as f:
        project_info = json.load(f)
        scenario_id = project_info['scenario_id']

    model_dict = dict((fd, fd_m.conf)for fd, fd_m in fd_metadata.items())
    top_k_model_dict = dict(
        sorted(model_dict.items(),
               key=itemgetter(1), reverse=True)[:MODEL_FDS_TOP_K])

    '''Compute the active learning '''
    overall_entropy_dict = compute_entropy_values(
        indices=candidate_unserved_indices,
        top_model_fds=top_k_model_dict,
        scenario_id=scenario_id)

    overall_violation_pairs = set()

    if resample:
        for fd in top_k_model_dict:
            overall_violation_pairs |= scenarios[scenario_id]['hypothesis_space'][fd]['violation_pairs']
    else:
        for fd in top_k_model_dict:
            overall_violation_pairs |= set([pair for pair in scenarios[scenario_id]['hypothesis_space'][fd]['violation_pairs'] if pair[0]
                                           in unserved_indices and pair[1] in unserved_indices and (pair[0] in candidate_unserved_indices or pair[1] in candidate_unserved_indices)])

    '''Vectorize the computation of the probability using numpy'''
    indices = list(overall_entropy_dict.keys())
    entropy_val = np.array([overall_entropy_dict[idx]
                            for idx in indices])
    sample_prob = np.exp(entropy_val/STOCHASTIC_UNCERTAINTY_SAMPLING_GAMMA)
    sample_prob = sample_prob/np.sum(sample_prob)

    '''Sample data using the sample probability'''
    s_out = set()
    sampled_indices = set(np.random.choice(
        indices, sample_size, replace=False, p=sample_prob))
    for idx in sampled_indices:
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
        missing_ones = [idx for idx in sampled_indices.difference(s_out)]
        s_out |= set(missing_ones[:remaining])

    return list(s_out)
