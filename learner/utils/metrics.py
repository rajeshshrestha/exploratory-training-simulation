import math
from operator import itemgetter
import numpy as np
import logging

from .initialize_variables import scenarios
from .initialize_variables import models_dict


logger = logging.getLogger(__file__)


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

    accuracy = np.mean(is_correct)
    recall = sum(true_positive)/(sum(is_dirty_true_labels)+1e-7)
    precision = sum(true_positive)/(sum(is_dirty_predicted_labels)+1e-7)
    f1 = 2*recall*precision/(recall+precision+1e-7)

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
