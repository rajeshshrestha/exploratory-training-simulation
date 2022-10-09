from .initialize_variables import models_dict


class FullOracleTrainer:
    def __init__(self, scenario_id, project_id, columns):
        super(FullOracleTrainer, self).__init__()

        self.scenario_id = scenario_id
        self.project_id = project_id
        self.columns = columns
        self.feedbackMap = None

    def update_feedback_map(self, data):
        # Decide for each row whether to mark or not
        for row in data.keys():
            if not models_dict[self.scenario_id]["predictions"][row]:
                for rh in self.columns:
                    self.feedbackMap[row][rh] = True
