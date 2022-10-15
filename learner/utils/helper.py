import json
import pickle
import logging

from .initialize_variables import scenarios
from .initialize_variables import models_dict
from .initialize_variables import validation_indices_dict
from .metrics import compute_metrics
from .sampling_policy import returnRandomTuples, returnActiveLearningTuples
from .sampling_policy import returnStochasticBRTuples
from .sampling_policy import returnStochasticActiveLRTuples
from .env_variables import MODEL_FDS_TOP_K


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
        open('./store/' + project_id + '/interaction_metadata.pk', 'rb'))
    # study_metrics = json.load( open('./store/' + project_id + '/study_metrics.json', 'r') )
    start_time = pickle.load(
        open('./store/' + project_id + '/start_time.pk', 'rb'))

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
        './store/' + project_id + '/interaction_metadata.pk', 'wb'))
    logger.info('*** Interaction metadata updates saved ***')


# Interpret user feedback and update alphas and betas for each FD in the hypothesis space


def interpretFeedback(s_in, feedback, project_id, current_iter,
                      current_time, trainer_model=None):
    fd_metadata = pickle.load(
        open('./store/' + project_id + '/fd_metadata.pk', 'rb'))
    start_time = pickle.load(
        open('./store/' + project_id + '/start_time.pk', 'rb'))
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
    mae_ground_model_error = 0
    mae_trainer_model_error = 0
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
            compliance_num = len([idx for idx in scenarios[scenario_id]
                                 ['hypothesis_space'][fd]['supports'].get(
                                     i, [])
                                 if idx in s_in.index])
            violation_num = len([idx for idx in scenarios[scenario_id]
                                 ['hypothesis_space']
                                 [fd]['violations'].get(i, [])
                                 if idx in s_in.index])

            if (compliance_num-violation_num) >= 0:  # tuple is clean
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
        mae_ground_model_error += abs(fd_m.conf -
                                      models_dict[scenario_id]['model'][fd])
        if trainer_model is not None:
            mae_trainer_model_error += abs(fd_m.conf - trainer_model[fd])

    '''Divide by number of fds'''
    mae_ground_model_error = mae_ground_model_error/len(fd_metadata)
    logger.info(f"MAE Ground Model Conf Error: {mae_ground_model_error}")

    if trainer_model is not None:
        mae_trainer_model_error = mae_trainer_model_error/len(fd_metadata)
        logger.info(f"MAE Trainer Model Conf Error: {mae_trainer_model_error}")

    '''Compute accuracy in Unserved dataset'''
    model_dict = dict((fd, fd_m.conf)for fd, fd_m in fd_metadata.items())
    val_idxs = validation_indices_dict[scenario_id]
    accuracy, recall, precision, f1 = compute_metrics(
        indices=val_idxs,
        model=model_dict,
        scenario_id=scenario_id,
        top_k=MODEL_FDS_TOP_K)
    logger.info(
        "=============================================================================")
    logger.info(
        f"Accuracy in {len(val_idxs)} unserved data: {round(accuracy,2)}")
    logger.info(
        "=============================================================================")
    study_metrics = json.load(
        open('./store/' + project_id + '/study_metrics.json', 'rb'))
    study_metrics['iter_accuracy'].append(accuracy)
    study_metrics['iter_recall'].append(recall)
    study_metrics['iter_precision'].append(precision)
    study_metrics['iter_f1'].append(f1)
    study_metrics['elapsed_time'].append(elapsed_time)
    study_metrics['iter_mae_ground_model_error'].append(mae_ground_model_error)
    if trainer_model is not None:
        study_metrics['iter_mae_trainer_model_error'].append(
            mae_trainer_model_error)

    json.dump(study_metrics,
              open('./store/' + project_id + '/study_metrics.json', 'w'))

    # Save updated alpha/beta metrics
    pickle.dump(fd_metadata, open(
        './store/' + project_id + '/fd_metadata.pk', 'wb'))


# Build a sample

def buildSample(data, sample_size, project_id,
                sampling_method,
                resample):
    if sampling_method == 'ACTIVELR':
        s_index = returnActiveLearningTuples(sample_size,
                                             project_id,
                                             resample=resample)
    elif sampling_method == 'RANDOM':
        s_index = returnRandomTuples(sample_size,
                                     project_id,
                                     resample=resample)
    elif sampling_method == 'STOCHASTICBR':
        s_index = returnStochasticBRTuples(sample_size=sample_size,
                                           project_id=project_id,
                                           resample=resample)
    elif sampling_method == 'STOCHASTICUS':
        s_index = returnStochasticActiveLRTuples(sample_size=sample_size,
                                                 project_id=project_id,
                                                 resample=resample)
    else:
        raise Exception(
            f"Unknown sampling method passed: {sampling_method}!!!")

    s_out = data.loc[s_index, :]

    logger.info(f'IDs of tuples in next sample: {s_out.index}')

    return s_out
