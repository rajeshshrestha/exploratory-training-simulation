from .full_oracle import FullOracleTrainer
from .bayesian import BayesianTrainer
from .learning_oracle import LearningOracleTrainer
import re


class TrainerModel:
    def __init__(self,
                 trainer_type,
                 trainer_prior_type,
                 scenario_id,
                 project_id,
                 columns) -> None:
        assert trainer_type in ['full-oracle',
                                'learning-oracle',
                                'bayesian'], f"Invalid trainer type: {trainer_type}"
        self.trainer_type = trainer_type

        assert trainer_prior_type in ['uniform-0.1',
                                      'uniform-0.5',
                                      'uniform-0.9',
                                      'data-estimate',
                                      'random'
                                      ], f"Invalid Trainer Prior:{trainer_prior_type} type used!!!"
        self.trainer_prior_type = trainer_prior_type

        if trainer_type == 'full-oracle':
            self.model = FullOracleTrainer(scenario_id=scenario_id,
                                           project_id=project_id,
                                           columns=columns)
        elif trainer_type == 'learning-oracle':
            self.model = LearningOracleTrainer(scenario_id=scenario_id,
                                               project_id=project_id,
                                               columns=columns,
                                               initial_p=0.1,
                                               p_step=0.05)
        elif trainer_type == 'bayesian':
            self.model = BayesianTrainer(scenario_id=scenario_id,
                                         project_id=project_id,
                                         columns=columns,
                                         prior_type=trainer_prior_type,
                                         p_max=0.9,
                                         top_k=10)

        '''Save columns on the data except the id column'''
        self.columns = columns

        self.scenario_id = scenario_id
        self.project_id = project_id

        '''Save initial model'''
        self.save_model()

    def save_model(self):
        self.model.save()

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
        self.save_model()

        feedback_dict = dict()
        for f in self.model.feedbackMap.keys():
            feedback_dict[data[f]['id']] = self.model.feedbackMap[f]

        return feedback_dict

    def get_model(self):
        return self.model.get_model_conf_dict()
