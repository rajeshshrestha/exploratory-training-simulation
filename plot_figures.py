# %% [markdown]
# ## Import libraries

# %%
import os
import matplotlib.pyplot as plt
import json
import os
from statistics import mean
import numpy as np

# %%


def read_metrics(run_folder_path):

    folders = os.listdir(run_folder_path)

    results_dict = dict((trainer_type, {
        'Random': {'iter_accuracy': [],
                   'iter_recall': [],
                   'iter_precision': [],
                   'iter_f1': [],
                   'iter_mae_ground_model_error': [],
                   'iter_mae_trainer_model_error': []
                   },
        'ActiveLR': {'iter_accuracy': [],
                     'iter_recall': [],
                     'iter_precision': [],
                     'iter_f1': [],
                     'iter_mae_ground_model_error': [],
                     'iter_mae_trainer_model_error': []
                     },
        'StochasticBR': {'iter_accuracy': [],
                         'iter_recall': [],
                         'iter_precision': [],
                         'iter_f1': [],
                         'iter_mae_ground_model_error': [],
                         'iter_mae_trainer_model_error': []
                         },
        'StochasticUS': {'iter_accuracy': [],
                         'iter_recall': [],
                         'iter_precision': [],
                         'iter_f1': [],
                         'iter_mae_ground_model_error': [],
                         'iter_mae_trainer_model_error': []
                         }
    }) for trainer_type in ['full-oracle', 'learning-oracle', 'bayesian'])

    for folder in folders:
        try:
            trainer_type, sampling_method, _ = folder.split("_")
            with open(os.path.join(run_folder_path, folder, "study_metrics.json"), 'r') as fp:
                study_metrics = json.load(fp)
                for metric_type in ['iter_accuracy', 'iter_recall', 'iter_precision', 'iter_f1', 'iter_mae_ground_model_error', 'iter_mae_trainer_model_error']:
                    folder_name = None
                    if sampling_method == 'RANDOM':
                        folder_name = 'Random'
                    elif sampling_method == 'ACTIVELR':
                        folder_name = 'ActiveLR'
                    elif sampling_method == 'STOCHASTICBR':
                        folder_name = 'StochasticBR'
                    elif sampling_method == 'STOCHASTICUS':
                        folder_name = 'StochasticUS'
                    results_dict[trainer_type][folder_name][metric_type].append([list(range(1, len(
                        study_metrics[metric_type])+1)), study_metrics['elapsed_time'], study_metrics[metric_type]])
        except Exception as e:
            print(e)
    return results_dict


def compute_average_metrics(results_dict):
    '''For Random'''
    average_dict = dict((trainer_type, {
        'Random': {'iter_accuracy': [],
                   'iter_recall': [],
                   'iter_precision': [],
                   'iter_f1': [],
                   'iter_mae_ground_model_error': [],
                   'iter_mae_trainer_model_error': []
                   },
        'ActiveLR': {'iter_accuracy': [],
                     'iter_recall': [],
                     'iter_precision': [],
                     'iter_f1': [],
                     'iter_mae_ground_model_error': [],
                     'iter_mae_trainer_model_error': []
                     },
        'StochasticBR': {'iter_accuracy': [],
                         'iter_recall': [],
                         'iter_precision': [],
                         'iter_f1': [],
                         'iter_mae_ground_model_error': [],
                         'iter_mae_trainer_model_error': []
                         },
        'StochasticUS': {'iter_accuracy': [],
                         'iter_recall': [],
                         'iter_precision': [],
                         'iter_f1': [],
                         'iter_mae_ground_model_error': [],
                         'iter_mae_trainer_model_error': []
                         }
    }) for trainer_type in ['full-oracle', 'learning-oracle', 'bayesian'])

    for trainer_type in ['full-oracle', 'learning-oracle', 'bayesian']:
        for sampling_type in ['Random', 'ActiveLR', 'StochasticBR', 'StochasticUS']:
            for metric_type in ['iter_accuracy', 'iter_recall', 'iter_precision', 'iter_f1', 'iter_mae_ground_model_error', 'iter_mae_trainer_model_error']:
                for exp_metrics_lst in zip(*results_dict[trainer_type][sampling_type][metric_type]):
                    average_lst = []
                    # print(exp_metrics_lst)

                    max_len = max(len(exp_metric)
                                  for exp_metric in exp_metrics_lst)
                    # print(max_len)
                    for idx in range(max_len):
                        candidate_lst = [exp_metric[idx] for exp_metric in exp_metrics_lst if idx < len(
                            exp_metric) and str(exp_metric[idx]) != 'nan']
                        if candidate_lst:
                            average_lst.append(mean(candidate_lst))
                        else:
                            average_lst.append(np.nan)
                    average_dict[trainer_type][sampling_type][metric_type].append(
                        average_lst)
    return average_dict


def plot_figures(run_folder_path):
    '''Read run folder'''
    results_dict = read_metrics(run_folder_path=run_folder_path)

    '''Compute average dict'''
    average_dict = compute_average_metrics(results_dict=results_dict)

    figures = []
    figure_names = []
    # for metric in ['accuracy', 'recall', 'precision', 'f1', 'mae_ground_model_error', 'mae_trainer_model_error']:
    for metric in ['mae_trainer_model_error']:
        for trainer_type in ['bayesian']:
            try:
                fig = plt.figure(figsize=(6, 4))
                for sampling_method in average_dict[trainer_type]:
                    plt.plot(average_dict[trainer_type][sampling_method]
                             [f'iter_{metric}'][0], average_dict[trainer_type][sampling_method][f'iter_{metric}'][2], label='US' if sampling_method == "ActiveLR" else sampling_method)
                plt.xlabel('Iterations')
                if metric == 'mae_ground_model_error':
                    plt.ylabel("Mean Absolute Error(MAE)")
                    # plt.title(
                    #     "Difference between Ground Model and Learner Model")
                elif metric == 'mae_trainer_model_error':
                    plt.ylabel("Mean Absolute Error(MAE)")
                    # plt.title(
                    #     "Difference between Trainer Model and Learner Model")
                else:
                    plt.ylabel(metric)
                    # plt.title(
                    #     "Predictions between the Learner and the Ground Truth Model")
                plt.ylim(bottom=0)
                plt.legend()
                plt.tight_layout()

                figures.append(fig)
                figure_names.append(f'{trainer_type}_{metric}.png')
            except Exception as e:
                print(e)

    return figures, figure_names


# %%
if __name__ == "__main__":
    base_project_dir = os.path.dirname(
        os.path.abspath(__file__))
    base_run_dir = os.path.join(base_project_dir, 'learner', 'store')
    fig_save_base_dir = os.path.join(base_project_dir, 'figures')
    os.makedirs(fig_save_base_dir, exist_ok=True)

    for dataset in ['airport', 'omdb']:
        for use_val_data in ['True', 'False']:
            dir1 = os.path.join(
                base_run_dir, f"dataset={dataset}", f"use_val_data={use_val_data}")
            if not os.path.exists(dir1):
                continue

            '''Loop into the dirty proportion folders'''
            proportion_folders = [fold for fold in os.listdir(
                dir1) if "dirty-proportion=" in fold]
            for prop_folder in proportion_folders:
                dir2 = os.path.join(dir1, prop_folder)
                dirty_proportion = prop_folder.replace("dirty-proportion=", "")

                '''Loop into the prior functions'''
                prior_folders = [fold for fold in os.listdir(
                    dir2) if "trainer-prior-type=" in fold and "learner-prior-type=" in fold]
                for prior_folder in prior_folders:
                    trainer_prior_type, learner_prior_type = prior_folder.replace(
                        "-learner-prior-type=", "_learner-prior-type=").split("_")
                    run_dir = os.path.join(dir2, prior_folder)
                    figures, figure_names = plot_figures(run_dir)

                    fig_save_dir = os.path.join(
                        fig_save_base_dir,
                        f"dataset={dataset}",
                        f"use_val_data={use_val_data}",
                        prop_folder,
                        f"{trainer_prior_type}_{learner_prior_type}")
                    os.makedirs(fig_save_dir, exist_ok=True)
                    for fig, fig_name in zip(figures, figure_names):
                        figure_save_path = os.path.join(fig_save_dir, fig_name)
                        print(f"Saving figure into {figure_save_path}...")
                        fig.savefig(figure_save_path)

                    plt.close('all')


# %%
