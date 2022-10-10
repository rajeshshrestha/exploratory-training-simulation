from .initialize_variables import models_dict
import random


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

    def update_feedback_map(self, data):
        # Decide for each row whether to mark or not
        for row in data.keys():
            is_prediction_correct = random.choices([True, False],
                                                   weights=[self.p_val,
                                                            1-self.p_val])
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
