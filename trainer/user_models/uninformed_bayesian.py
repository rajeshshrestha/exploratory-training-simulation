from .initialize_variables import required_fds
from .initialize_variables import scenarios
import math
# import numpy as np
from operator import itemgetter
import logging
import random
from statistics import mean
import os
import json

logger = logging.getLogger(__file__)
current_path = os.path.dirname(os.path.realpath(__file__))


class FDMeta:
    """
    Build the feedback dictionary object that will be utilized during the 
    interaction
    """

    def __init__(self, fd, alpha, beta):
        self.fd = fd
        self.lhs = fd.split(' => ')[0][1:-1].split(', ')
        self.rhs = fd.split(' => ')[1].split(', ')
        self.alpha = alpha
        self.alpha_history = [alpha]
        self.beta = beta
        self.beta_history = [beta]
        self.conf = (alpha / (alpha+beta))

    def to_dict(self):
        return {
            'fd': self.fd,
            'lhs': self.lhs,
            'rhs': self.rhs,
            'alpha': self.alpha,
            'beta': self.beta,
            'conf': self.conf,
            'alpha_history': self.alpha_history,
            'beta_history': self.beta_history
        }


class UninformedBayesianTrainer:
    def __init__(self, scenario_id, project_id, columns, p_max, alpha, beta, top_k) -> None:

        self.scenario_id = scenario_id
        self.project_id = project_id
        self.columns = columns
        self.top_k = top_k
        self.feedbackMap = None

        '''Probability to stick with prediction from model'''
        self.p_max = p_max

        self.fd_metadata = dict((fd, FDMeta(fd=fd, alpha=alpha, beta=beta))
                                for fd in required_fds[self.scenario_id])

        '''Create directory for storing model'''
        self.trainer_store_path = os.path.join(os.path.dirname(
            current_path), "trainer-store", self.project_id)
        os.makedirs(self.trainer_store_path)

    def get_model_dict(self):
        return dict((fd, fd_metadata.to_dict())
                    for fd, fd_metadata in self.fd_metadata.items())

    def save(self):
        with open(os.path.join(self.trainer_store_path,
                               'model.json'), 'w') as fp:
            json.dump(self.get_model_dict(), fp)

    @staticmethod
    # Calculate the initial probability mean for the FD
    def initialPrior(mu, variance):
        beta = (1 - mu) * ((mu * (1 - mu) / variance) - 1)
        alpha = (mu * beta) / (1 - mu)
        return alpha, beta

    def update_model(self, data):
        for fd, fd_m in self.fd_metadata.items():
            data_idxs = list(data.keys())
            for i in range(len(data_idxs)-1):
                for j in range(i, len(data_idxs)):
                    if data_idxs[j] in scenarios[self.scenario_id]['hypothesis_space'][fd]['violations'].get(data_idxs[i], set()):
                        fd_m.beta += 1
                    else:
                        fd_m.alpha += 1

            fd_m.alpha_history.append(fd_m.alpha)
            fd_m.beta_history.append(fd_m.beta)
            fd_m.conf = fd_m.alpha/(fd_m.alpha + fd_m.beta)

        # for fd, fd_m in self.fd_metadata.items():
        #     print(fd, fd_m.conf, fd_m.alpha, fd_m.beta)

    @staticmethod
    def compute_conditional_clean_prob(idx, fd, fd_prob, scenario_id, data_indices=None):
        if data_indices is None:
            compliance_num = len(
                scenarios[scenario_id]['hypothesis_space'][fd]['supports'].get(idx, []))
            violation_num = len(
                scenarios[scenario_id]['hypothesis_space'][fd]['violations'].get(idx, []))
        else:
            compliance_num = len([idx_ for idx_ in scenarios[scenario_id]['hypothesis_space']
                                  [fd]['supports'].get(idx, [])
                                  if idx_ in data_indices])
            violation_num = len([idx_ for idx_ in scenarios[scenario_id]['hypothesis_space']
                                [fd]['violations'].get(idx, []) if idx_ in data_indices])

        tuple_clean_score = math.exp(fd_prob*(compliance_num-violation_num))
        tuple_dirty_score = math.exp(fd_prob*(-compliance_num+violation_num))
        cond_p_clean = tuple_clean_score/(tuple_clean_score+tuple_dirty_score)

        return cond_p_clean

    @classmethod
    def get_average_cond_clean_prediction(cls, indices, model, scenario_id):
        conditional_clean_probability_dict = dict()
        indices = set(indices)
        for idx in indices:
            cond_clean_prob = mean([cls.compute_conditional_clean_prob(
                idx=idx, fd=fd, fd_prob=fd_prob, scenario_id=scenario_id,
                data_indices=indices) for fd, fd_prob in model.items()])  # whether to include the validation_indices or all the data_indices while computing the conditional clean probability
            conditional_clean_probability_dict[idx] = cond_clean_prob
        return conditional_clean_probability_dict

    def get_predictions(self, indices):

        model_dict = dict((fd, fd_m.conf)
                          for fd, fd_m in self.fd_metadata.items())

        '''Pick top k fds if needed'''
        if 0 < self.top_k < len(model_dict):
            model = dict(sorted(model_dict.items(), key=itemgetter(1),
                                reverse=True)[:self.top_k])

        conditional_clean_probability_dict = \
            self.get_average_cond_clean_prediction(
                indices=indices,
                model=model,
                scenario_id=self.scenario_id)
        logger.info(conditional_clean_probability_dict)

        idxs = list(conditional_clean_probability_dict.keys())
        is_dirty_predicted_dict = \
            dict(
                (idx, conditional_clean_probability_dict[idx] < 0.5)
                for idx in idxs)
        return is_dirty_predicted_dict

    def update_feedback_map(self, data):
        self.update_model(data)

        is_dirty_predicted_dict = self.get_predictions(indices=data.keys())

        logger.info(f"Previous feedback_dict: {self.feedbackMap}")

        for row in data.keys():
            stick_with_prediction = random.choices([True, False],
                                                   weights=[self.p_max,
                                                            1-self.p_max])[0]
            if stick_with_prediction:
                '''Mark dirty if predicted dirty based on the model'''
                if is_dirty_predicted_dict[row]:
                    for rh in self.columns:
                        self.feedbackMap[row][rh] = True
            else:
                '''Mark randomly'''
                label = random.choices([True, False])[0]
                for rh in self.columns:
                    self.feedbackMap[row][rh] = label

        logger.info(f"Later feedback_dict: {self.feedbackMap}")
