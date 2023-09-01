# %% [markdown]
# ## Import libraries

# %%
import os
import matplotlib.pyplot as plt
import json
import os
from statistics import mean
import numpy as np
import shutil
# %%


def read_metrics(run_folder_path):

    folders = os.listdir(run_folder_path)

    results_dict = dict((trainer_type, {
        'Random': {'iter_accuracy': [],
                   'iter_recall': [],
                   'iter_precision': [],
                   'iter_f1': [],
                   'iter_accuracy_converged': [],
                   'iter_recall_converged': [],
                   'iter_precision_converged': [],
                   'iter_f1_converged': [],
                   'iter_mae_ground_model_error': [],
                   'iter_mae_trainer_model_error': []
                   },
        'ActiveLR': {'iter_accuracy': [],
                     'iter_recall': [],
                     'iter_precision': [],
                     'iter_f1': [],
                     'iter_accuracy_converged': [],
                     'iter_recall_converged': [],
                     'iter_precision_converged': [],
                     'iter_f1_converged': [],
                     'iter_mae_ground_model_error': [],
                     'iter_mae_trainer_model_error': []
                     },
        'StochasticBR': {'iter_accuracy': [],
                         'iter_recall': [],
                         'iter_precision': [],
                         'iter_f1': [],
                         'iter_accuracy_converged': [],
                         'iter_recall_converged': [],
                         'iter_precision_converged': [],
                         'iter_f1_converged': [],
                         'iter_mae_ground_model_error': [],
                         'iter_mae_trainer_model_error': []
                         },
        'StochasticUS': {'iter_accuracy': [],
                         'iter_recall': [],
                         'iter_precision': [],
                         'iter_f1': [],
                         'iter_accuracy_converged': [],
                         'iter_recall_converged': [],
                         'iter_precision_converged': [],
                         'iter_f1_converged': [],
                         'iter_mae_ground_model_error': [],
                         'iter_mae_trainer_model_error': []
                         }
    }) for trainer_type in ['full-oracle', 'learning-oracle', 'bayesian'])

    for folder in folders:
        try:
            trainer_type, sampling_method, _ = folder.split("_")
            with open(os.path.join(run_folder_path, folder, "study_metrics.json"), 'r') as fp:
                study_metrics = json.load(fp)
                for metric_type in ['iter_accuracy',
                                    'iter_recall',
                                    'iter_precision',
                                    'iter_f1',
                                    'iter_mae_ground_model_error',
                                    'iter_mae_trainer_model_error',
                                    'iter_accuracy_converged',
                                    'iter_recall_converged',
                                    'iter_precision_converged',
                                    'iter_f1_converged']:
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
                   'iter_accuracy_converged': [],
                   'iter_recall_converged': [],
                   'iter_precision_converged': [],
                   'iter_f1_converged': [],
                   'iter_mae_ground_model_error': [],
                   'iter_mae_trainer_model_error': []
                   },
        'ActiveLR': {'iter_accuracy': [],
                     'iter_recall': [],
                     'iter_precision': [],
                     'iter_f1': [],
                     'iter_accuracy_converged': [],
                     'iter_recall_converged': [],
                     'iter_precision_converged': [],
                     'iter_f1_converged': [],
                     'iter_mae_ground_model_error': [],
                     'iter_mae_trainer_model_error': []
                     },
        'StochasticBR': {'iter_accuracy': [],
                         'iter_recall': [],
                         'iter_precision': [],
                         'iter_f1': [],
                         'iter_accuracy_converged': [],
                         'iter_recall_converged': [],
                         'iter_precision_converged': [],
                         'iter_f1_converged': [],
                         'iter_mae_ground_model_error': [],
                         'iter_mae_trainer_model_error': []
                         },
        'StochasticUS': {'iter_accuracy': [],
                         'iter_recall': [],
                         'iter_precision': [],
                         'iter_f1': [],
                         'iter_accuracy_converged': [],
                         'iter_recall_converged': [],
                         'iter_precision_converged': [],
                         'iter_f1_converged': [],
                         'iter_mae_ground_model_error': [],
                         'iter_mae_trainer_model_error': []
                         }
    }) for trainer_type in ['full-oracle', 'learning-oracle', 'bayesian'])

    for trainer_type in ['full-oracle', 'learning-oracle', 'bayesian']:
        for sampling_type in ['Random', 'ActiveLR', 'StochasticBR', 'StochasticUS']:
            for metric_type in ['iter_accuracy', 'iter_recall',
                                'iter_precision', 'iter_f1',
                                'iter_accuracy_converged',
                                'iter_recall_converged',
                                'iter_precision_converged',
                                'iter_f1_converged',
                                'iter_mae_ground_model_error',
                                'iter_mae_trainer_model_error']:
                try:
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
                except Exception as e:
                    print(e)
    return average_dict


def print_metrics(run_folder_path, print_intervals=[0, 0.1, 0.5, 0.9, 1.0]):
    try:
        '''Read run folder'''
        results_dict = read_metrics(run_folder_path=run_folder_path)

        '''Compute average dict'''
        average_dict = compute_average_metrics(results_dict=results_dict)

        prior_type = run_folder_path.split("/")[-1]
        print("*****************************************************************************")
        print(f"Prior Type: {prior_type}")
        print("*****************************************************************************")

        for trainer_type in ['bayesian']:
            for metric in ['accuracy_converged',
                        'recall_converged',
                        'precision_converged',
                        'f1_converged']:
                print(
                    f"Trainer Type: {trainer_type} Metric: {metric}")
                for interval in print_intervals:
                    print_str = "Iteration Fraction: %.2f "%round(interval,2)
                    for sampling_method in average_dict[trainer_type]:
                        total_iter = len(
                            average_dict[trainer_type][sampling_method][f'iter_{metric}'][0])-1
                        
                        if total_iter < 0:
                            continue
                        
                        iter = int(total_iter*interval)
                        print_str += f"{sampling_method}: %.2f "%round(average_dict[trainer_type][sampling_method][f'iter_{metric}'][2][iter], 3)
                    print(print_str)
                print("-------------------------------------------------------------------")
        print("========================================================================")
    except Exception as e:
        print(e)


def plot_figures(run_folder_path):
    '''Read run folder'''
    results_dict = read_metrics(run_folder_path=run_folder_path)

    '''Compute average dict'''
    average_dict = compute_average_metrics(results_dict=results_dict)

    figures = []
    figure_names = []
    for metric in ['accuracy',
                   'recall',
                   'precision',
                   'f1',
                   'accuracy_converged',
                   'recall_converged',
                   'precision_converged',
                   'f1_converged',
                   'mae_ground_model_error',
                   'mae_trainer_model_error']:
        # for metric in ['mae_trainer_model_error', 'mae_ground_model_error']:
        for trainer_type in ['bayesian']:
            fig = plt.figure(figsize=(6, 4))
            for sampling_method in average_dict[trainer_type]:
                if len(average_dict[trainer_type][sampling_method][f'iter_{metric}']) != 0 and len(average_dict[trainer_type][sampling_method][f'iter_{metric}'][0]) != 0 and len(average_dict[trainer_type][sampling_method][f'iter_{metric}'][2]) !=0:
                    x_vals = average_dict[trainer_type][sampling_method][f'iter_{metric}'][0]
                    y_vals = average_dict[trainer_type][sampling_method][f'iter_{metric}'][2]

                    # if len(x_vals) > 30:
                    #     x_vals = x_vals[:30]
                    
                    # if len(y_vals) > 30:
                    #     y_vals = y_vals[:30]

                    plt.plot(x_vals, y_vals, label='US' if sampling_method == "ActiveLR" else sampling_method)
            plt.xlabel('Iterations')
            if metric == 'mae_ground_model_error':
                plt.ylabel("Mean Absolute Error(MAE)")
                # plt.title(
                #     "Difference between Ground Model and Learner Model")
            elif metric == 'mae_trainer_model_error':
                plt.ylabel("Mean Absolute Error(MAE)")
                # plt.title(
                #     "Difference between Trainer Model and Learner Model")
            elif "f1" in metric:
                plt.ylabel("F1 Score")
            elif "precision" in metric:
                plt.ylabel("Precision")
            elif "recall" in metric:
                plt.ylabel("Recall")
            else:
                plt.ylabel(metric)
                # plt.title(
                #     "Predictions between the Learner and the Ground Truth Model")
            plt.ylim(bottom=0)
            plt.legend()
            plt.tight_layout()

            figures.append(fig)
            figure_names.append(f'{trainer_type}_{metric}.png')

    return figures, figure_names


# %%
if __name__ == "__main__":
    project_name = os.getenv("PROJECT_NAME", None)
    base_project_dir = os.path.dirname(
        os.path.abspath(__file__))
    if project_name is None:
        base_run_dir = os.path.join(base_project_dir, 'learner', 'store')
        fig_save_base_dir = os.path.join(base_project_dir, 'figures')

    else:
        base_run_dir = os.path.join("/data/shresthr", project_name, "store")
        fig_save_base_dir = os.path.join("/data/shresthr", project_name, "figures")
        data_save_base_dir = os.path.join("/data/shresthr", project_name, "data")

        '''Create data directory'''
        if os.path.exists(data_save_base_dir):
            shutil.rmtree(data_save_base_dir)
        os.makedirs(data_save_base_dir)

        '''Copy new_scenarios, trainer_model and preprocessed files'''
        shutil.copyfile("./scenarios.json", os.path.join("/data/shresthr", project_name,"data/scenarios.json"))
        shutil.copyfile("./new_scenarios.json", os.path.join("/data/shresthr", project_name,"data/new_scenarios.json"))
        shutil.copyfile("./trainer_model.json", os.path.join("/data/shresthr", project_name,"data/trainer_model.json"))
        shutil.copytree("./data/preprocessed-data", os.path.join("/data/shresthr", project_name,"data/preprocessed-data"))
    
    os.makedirs(fig_save_base_dir, exist_ok=True)

    for dataset in ['airport', 'omdb', 'hospital', 'tax']:
        print(dataset)
        for use_val_data in ['True', 'False']:
            dir1 = os.path.join(
                base_run_dir, f"dataset={dataset}", f"use_val_data={use_val_data}")
            if not os.path.exists(dir1):
                print(f"Directory doesn't exist: {base_run_dir}")
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
                    print_metrics(run_dir)

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
