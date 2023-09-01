import os
from .initialize_variables import models_dict
import random
import json

current_path = os.path.dirname(os.path.realpath(__file__))


class LearningOracleTrainer:
    def __init__(self, scenario_id, project_id, columns, initial_p, p_step):
        super(LearningOracleTrainer, self).__init__()

        self.scenario_id = scenario_id
        self.project_id = project_id
        self.columns = columns
        self.feedbackMap = None

        '''Initial correct probability and step size for its improvement'''
        self.p_val = initial_p
        self.p_step = p_step
        self.p_val_history = [self.p_val]

        '''Create directory for storing model'''
        self.trainer_store_path = os.path.join(os.path.dirname(
            current_path), "trainer-store", self.project_id)
        os.makedirs(self.trainer_store_path)

    def update_feedback_map(self, data):
        for row in data.keys():
            is_prediction_correct = random.choices([True, False],
                                                   weights=[self.p_val,
                                                            1-self.p_val])[0]
            if is_prediction_correct:
                '''Mark dirty if not found clean in the model dict'''
                if not models_dict[self.scenario_id]["predictions"][row]:
                    for rh in self.columns:
                        self.feedbackMap[row][rh] = True
            else:
                '''Mark dirty if found as clean in the model dict'''
                if models_dict[self.scenario_id]["predictions"][row]:
                    for rh in self.columns:
                        self.feedbackMap[row][rh] = True

        '''Update p_val'''
        self.p_val = min(self.p_val+self.p_step, 1.0)
        self.p_val_history.append(self.p_val)

    def get_model_dict(self):
        return {'p_val': self.p_val,
                'p_step': self.p_step,
                'p_val_history': self.p_val_history
                }

    def get_model_conf_dict(self):
        return models_dict[self.scenario_id]["model"]

    def save(self):
        with open(os.path.join(self.trainer_store_path, 'model.json'),
                  'w') as fp:
            json.dump(self.get_model_dict(), fp)
