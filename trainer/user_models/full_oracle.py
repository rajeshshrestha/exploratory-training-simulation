import os
from .initialize_variables import models_dict

current_path = os.path.dirname(os.path.realpath(__file__))


class FullOracleTrainer:
    def __init__(self, scenario_id, project_id, columns):
        super(FullOracleTrainer, self).__init__()

        self.scenario_id = scenario_id
        self.project_id = project_id
        self.columns = columns
        self.feedbackMap = None

        '''Create directory for storing model'''
        self.trainer_store_path = os.path.join(os.path.dirname(
            current_path), "trainer-store", self.project_id)
        os.makedirs(self.trainer_store_path)

    def update_feedback_map(self, data):
        # Decide for each row whether to mark or not
        for row in data.keys():
            if not models_dict[self.scenario_id]["predictions"][row]:
                for rh in self.columns:
                    self.feedbackMap[row][rh] = True

    def get_model_conf_dict(self):
        return models_dict[self.scenario_id]["model"]

    def save(self):
        '''Nothing to do here'''
        pass
