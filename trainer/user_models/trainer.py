from .full_oracle import FullOracleTrainer
from .uninformed_bayesian import UninformedBayesianTrainer
import re


class TrainerModel:
    def __init__(self, trainer_type, scenario_id, project_id, columns, **kwargs) -> None:
        assert trainer_type in ['full-oracle', 'uninformed-bayesian']

        if trainer_type == 'full-oracle':
            self.model = FullOracleTrainer(scenario_id=scenario_id,
                                           project_id=project_id,
                                           columns=columns)
        elif trainer_type == 'uninformed-bayesian':
            self.model = UninformedBayesianTrainer(scenario_id=scenario_id,
                                                   project_id=project_id,
                                                   columns=columns)

        '''Save columns on the data except the id column'''
        self.columns = columns

        self.scenario_id = scenario_id
        self.project_id = project_id

    def buildFeedbackMap(self, data, feedback):
        feedbackMap = dict()
        for row in data.keys():
            tup = dict()
            for col in self.columns:
                trimmedCol = re.sub(r'/[\n\r]+/g', '', col)
                cell = next(f for f in feedback if
                            f['row'] == data[row]['id'] and f[
                                'col'] == trimmedCol)
                tup[col] = cell['marked']
            feedbackMap[row] = tup

        self.model.feedbackMap = feedbackMap
        return feedbackMap

    def get_feedback_dict(self, data, feedback):

        self.buildFeedbackMap(data=data, feedback=feedback)
        self.model.update_feedback_map(data=data)

        feedback_dict = dict()
        for f in self.model.feedbackMap.keys():
            feedback_dict[data[f]['id']] = self.model.feedbackMap[f]

        return feedback_dict
