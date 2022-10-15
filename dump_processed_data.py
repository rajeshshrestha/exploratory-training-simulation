# %% [markdown]
# ## Import libraries

# %%
from random import sample
from statistics import mean
from copy import deepcopy
import pandas as pd
import json
from tqdm import tqdm
import os
import numpy as np
import math
from operator import itemgetter
import pickle as pk
from tqdm import tqdm
import argparse

# %%
'''Parse arguments'''
parser = argparse.ArgumentParser()
parser.add_argument('--max-clean-num', default=1500, type=int)
parser.add_argument('--max-dirty-prop', default=0.1, type=float)
parser.add_argument('--dataset', default='airport', type=str)

args = parser.parse_args()
print(args)


clean_max_num = args.max_clean_num
dirty_sample_percentage = args.max_dirty_prop

DATASET = args.dataset
SCENARIO_ID = "3" if DATASET == "omdb" else "11"

# %% [markdown]
# ## Read raw data

# %%
data_path = f"./data/raw-data/{DATASET}-clean-full.csv"

raw_df = pd.read_csv(data_path)
raw_df.rename(columns=dict((col, col.lower())
              for col in raw_df.columns), inplace=True)
raw_df.index = raw_df.index.map(str)

# %%
# Read Scenario3 info for the functional dependencies

with open("./scenarios.json", 'r') as fp:
    scenarios = json.load(fp)
required_scenario_info = scenarios[SCENARIO_ID]
hypothesis_space = [hypothesis['cfd']
                    for hypothesis in required_scenario_info['hypothesis_space']]

# %%
with open("./new_scenarios.json", 'r') as fp:
    new_scenarios_dict = json.load(fp)

# %%
hypothesis_support_violation_ratio_info = dict()
for hypothesis in new_scenarios_dict[DATASET]['hypothesis_space']:
    hypothesis_info_dict = new_scenarios_dict[DATASET]['hypothesis_space'][hypothesis]
    if len(hypothesis_info_dict['lfd']+hypothesis_info_dict['rfd']) not in [3, 4]:
        continue

    support_pairs_num, violation_pairs_num = 0, 0
    for idx in hypothesis_info_dict['supports']:
        support_pairs_num += len(hypothesis_info_dict['supports'][idx])

    for idx in hypothesis_info_dict['violations']:
        violation_pairs_num += len(hypothesis_info_dict['violations'][idx])

    hypothesis_support_violation_ratio_info[hypothesis] = support_pairs_num/(
        support_pairs_num+violation_pairs_num+1e-7)


# %%
'''Sample confidence from 0 to support_violation_ratio'''
np.random.seed(1000)
model = dict((hypothesis, np.random.uniform(max(0, ratio-0.25), min(1, ratio+0.25)))
             for hypothesis, ratio in hypothesis_support_violation_ratio_info.items())

if os.path.exists("./trainer_model.json"):
    with open("./trainer_model.json", 'r') as fp:
        model_dict = json.load(fp)
        model_dict[DATASET] = {'model': model}
else:
    model_dict = {DATASET: {'model': model}}

# %%


def predict_clean_tuple(idx, model):
    total_score = 0
    for hypothesis, conf in model.items():
        '''Check the number of supports and violations based on the model'''
        support_pairs_num = len(
            new_scenarios_dict[DATASET]['hypothesis_space'][hypothesis]['supports'].get(idx, []))
        violation_pairs_num = len(
            new_scenarios_dict[DATASET]['hypothesis_space'][hypothesis]['violations'].get(idx, []))

        '''Vote according to the hypothesis conf'''
        total_score += (conf*(support_pairs_num-violation_pairs_num))

    if total_score > 0:
        return True, total_score

    else:
        return False, total_score


# %%
clean_tuple_indices = set()
model_score_dict = dict()
for idx in new_scenarios_dict[DATASET]['data_indices']:
    is_clean, total_score = predict_clean_tuple(idx, model)
    if is_clean:
        clean_tuple_indices.add(idx)
    model_score_dict[idx] = total_score


# %%
len(clean_tuple_indices)

# %%
model_dict[DATASET]['predictions'] = dict((idx, True) if idx in clean_tuple_indices else (
    idx, False) for idx in new_scenarios_dict[DATASET]['data_indices'])

# %%
with open("./trainer_model.json", 'w') as fp:
    json.dump(model_dict, fp)

# %%


def get_conditional_clean_prob(idx, fd, model_probab, valid_indices=None):
    if valid_indices is None:
        compliance_num = len(
            new_scenarios_dict[DATASET]['hypothesis_space'][fd]['supports'].get(str(idx), []))
        violation_num = len(
            new_scenarios_dict[DATASET]['hypothesis_space'][fd]['violations'].get(str(idx), []))
    else:
        compliance_num = len([idx_ for idx_ in new_scenarios_dict[DATASET]['hypothesis_space']
                             [fd]['supports'].get(str(idx), []) if idx_ in valid_indices])
        violation_num = len([idx_ for idx_ in new_scenarios_dict[DATASET]['hypothesis_space']
                            [fd]['violations'].get(str(idx), []) if idx_ in valid_indices])

    tuple_clean_score = math.exp(model_probab*(compliance_num-violation_num))
    tuple_dirty_score = math.exp(model_probab*(-compliance_num+violation_num))
    cond_p_clean = tuple_clean_score/(tuple_clean_score+tuple_dirty_score)
    return cond_p_clean


# %%
model = model_dict[DATASET]['model']
conditional_clean_probability_dict = dict()
clean_indices = set()
dirty_indices = set()

data_indices = new_scenarios_dict[DATASET]['data_indices']

top_10_fds = dict(sorted(model.items(), key=itemgetter(1), reverse=True)[:10])

for idx in data_indices:
    conditional_clean_probability_dict[idx] = {'hypothesis': dict()}
    for fd, model_probab in top_10_fds.items():
        conditional_clean_probability_dict[idx]['hypothesis'][fd] = get_conditional_clean_prob(
            idx, fd, model_probab=model_probab)
    conditional_clean_probability_dict[idx]['average'] = np.mean(
        list(conditional_clean_probability_dict[idx]['hypothesis'].values()))
    is_idx_clean = conditional_clean_probability_dict[idx]['average'] >= 0.5
    conditional_clean_probability_dict[idx]['is_clean'] = is_idx_clean

    if is_idx_clean:
        clean_indices.add(idx)
    else:
        dirty_indices.add(idx)
else:
    # pprint(conditional_clean_probability_dict)
    print(len(clean_indices), len(dirty_indices))

clean_sample_idxs = np.random.choice(list(clean_indices), min(
    len(clean_indices), clean_max_num), replace=False)
dirty_sample_idxs = np.random.choice(list(dirty_indices), int(
    dirty_sample_percentage*len(clean_sample_idxs)), replace=False)
sampled_data_indices = set(clean_sample_idxs).union(set(dirty_sample_idxs))
print(len(sampled_data_indices))


# %% [markdown]
# ## Rerun the model computation and is_clean prediction using this computed model

# %%

# %%
'''Assume every data to be clean at the beginning and compute the model based on that'''
new_model = deepcopy(model_dict[DATASET]['model'])
model_mae = float("inf")

new_data_indices = sampled_data_indices

while model_mae > 1e-05:

    '''Use current model to predict clean and dirty indices'''
    top_10_fds = dict(
        sorted(new_model.items(), key=itemgetter(1), reverse=True)[:10])

    new_conditional_clean_probability_dict = dict()
    new_clean_indices = set()
    new_dirty_indices = set()

    for idx in new_data_indices:
        new_conditional_clean_probability_dict[idx] = {'hypothesis': dict()}
        for fd, model_probab in top_10_fds.items():
            new_conditional_clean_probability_dict[idx]['hypothesis'][fd] = get_conditional_clean_prob(
                idx, fd, model_probab=model_probab, valid_indices=new_data_indices)
        new_conditional_clean_probability_dict[idx]['average'] = np.mean(
            list(new_conditional_clean_probability_dict[idx]['hypothesis'].values()))
        is_idx_clean = new_conditional_clean_probability_dict[idx]['average'] >= 0.5
        new_conditional_clean_probability_dict[idx]['is_clean'] = is_idx_clean

        if is_idx_clean:
            new_clean_indices.add(idx)
        else:
            new_dirty_indices.add(idx)

    else:
        # pprint(new_conditional_clean_probability_dict)
        print(f"Clean Data Number: {len(new_clean_indices)},"
              f"Dirty Data Number: {len(new_dirty_indices)},"
              f"Dirty Data Proportion: {len(new_dirty_indices)/len(new_clean_indices.union(new_dirty_indices))}")

    '''Use clean data to estimate model'''
    model_mae = 0
    for hypothesis in new_scenarios_dict[DATASET]['hypothesis_space']:
        hypothesis_info_dict = new_scenarios_dict[DATASET]['hypothesis_space'][hypothesis]
        if len(hypothesis_info_dict['lfd']+hypothesis_info_dict['rfd']) not in [3, 4]:
            continue

        '''Only consider the clean estimated indices'''
        support_pairs_num, violation_pairs_num = 0, 0
        for idx in hypothesis_info_dict['supports']:
            if idx not in new_clean_indices:
                continue
            support_pairs_num += len([idx1 for idx1 in hypothesis_info_dict['supports']
                                     [idx] if idx1 in new_clean_indices])

        for idx in hypothesis_info_dict['violations']:
            if idx not in new_clean_indices:
                continue
            violation_pairs_num += len(
                [idx1 for idx1 in hypothesis_info_dict['violations'][idx] if idx1 in new_clean_indices])

        fd_prob = support_pairs_num / \
            (support_pairs_num+violation_pairs_num+1e-7)

        '''Compute mae with previous model value'''
        model_mae += abs(new_model[hypothesis]-fd_prob)
        new_model[hypothesis] = fd_prob

    print(f"MAE: {model_mae}")

final_dirty_proportion = len(new_dirty_indices) / \
    len(new_clean_indices.union(new_dirty_indices))
print(f"Final Proportion: {final_dirty_proportion}")
# %%
model_dict[DATASET] = {'model': new_model,
                       'dirty_proportion': final_dirty_proportion}
model_dict[DATASET]['predictions'] = dict(
    (idx, True) if idx in new_clean_indices else (idx, False) for idx in new_data_indices)
with open("./trainer_model.json", 'w') as fp:
    json.dump(model_dict, fp)

# %%
sampled_df = raw_df.loc[list(new_data_indices)]
sampled_df

# %%
os.makedirs("./data/processed-data", exist_ok=True)
sampled_df.to_csv(f"./data/processed-data/{DATASET}-sampled.csv")

# %% [markdown]
# ## Final Process and Dump pickled data

# %%
with open('./trainer_model.json', 'r') as f:
    models_dict = json.load(f)

required_fds = dict(
    (scenario, set(models_dict[scenario]['model'].keys())) for scenario in models_dict)

with open("./data/processed-exp-data/trainer_model.json", 'w') as fp:
    json.dump(models_dict, fp)

with open("./data/processed-exp-data/required_fds.pk", 'wb') as fp:
    pk.dump(required_fds, fp)


with open('./new_scenarios.json', 'r') as f:
    scenarios = json.load(f)

'''Process new_scenarios to make the processing faster later'''
processed_df = dict()
filtered_processed_scenarios = dict()
for dataset in models_dict:
    processed_df[dataset] = pd.read_csv(
        scenarios[dataset]['processed_dataset_path'], index_col=0)
    processed_df[dataset].index = processed_df[dataset].index.map(str)
    required_indices = set(processed_df[dataset].index)

    filtered_processed_scenarios[dataset] = {'data_indices': set(
        scenarios[dataset]['data_indices']).intersection(required_indices), 'hypothesis_space': dict()}

    '''Filter required fds and data_indices'''
    for hypothesis in tqdm(scenarios[dataset]['hypothesis_space']):
        if hypothesis not in required_fds[dataset]:
            continue

        filtered_processed_scenarios[dataset]['hypothesis_space'][hypothesis] = {'lfd': set(
            scenarios[dataset]['hypothesis_space'][hypothesis]['lfd']),
            'rfd': set(
            scenarios[dataset]['hypothesis_space'][hypothesis]['rfd'])}

        for info_type in ['supports', 'violations']:
            filtered_processed_scenarios[dataset]['hypothesis_space'][hypothesis][info_type] = dict(
            )

            filtered_processed_scenarios[dataset]['hypothesis_space'][
                hypothesis][f'{info_type[:-1]}_pairs'] = set()
            for idx in scenarios[dataset]['hypothesis_space'][hypothesis][info_type]:
                if idx not in required_indices:
                    continue

                filtered_processed_scenarios[dataset]['hypothesis_space'][hypothesis][info_type][idx] = set(
                    scenarios[dataset]['hypothesis_space'][hypothesis][info_type][idx]).intersection(required_indices)

                pairs = set((idx, idx_) if idx < idx_ else (
                    idx_, idx) for idx_ in filtered_processed_scenarios[dataset]['hypothesis_space'][hypothesis][info_type][idx])
                filtered_processed_scenarios[dataset]['hypothesis_space'][
                    hypothesis][f'{info_type[:-1]}_pairs'] |= pairs


with open("./data/processed-exp-data/filtered_processed_scenarios.pk", 'wb') as fp:
    pk.dump(filtered_processed_scenarios, fp)

with open("./data/processed-exp-data/processed_dfs.pk", 'wb') as fp:
    pk.dump(processed_df, fp)


# %% [markdown]
# ## Final Validation

# %%

# %%
with open('./trainer_model.json', 'r') as f:
    _models_dict = json.load(f)

with open("./data/processed-exp-data/filtered_processed_scenarios.pk", 'rb') as fp:
    _filtered_processed_scenarios = pk.load(fp)

with open("./data/processed-exp-data/processed_dfs.pk", 'rb') as fp:
    _processed_df = pk.load(fp)


# %%
def compute_conditional_clean_prob(idx, fd, fd_prob, scenario_id, data_indices=None):
    if data_indices is None:
        compliance_num = len(
            _filtered_processed_scenarios[scenario_id]['hypothesis_space'][fd]['supports'].get(idx, []))
        violation_num = len(
            _filtered_processed_scenarios[scenario_id]['hypothesis_space'][fd]['violations'].get(idx, []))
    else:
        compliance_num = len([idx_ for idx_ in _filtered_processed_scenarios[scenario_id]['hypothesis_space']
                              [fd]['supports'].get(idx, [])
                              if idx_ in data_indices])
        violation_num = len([idx_ for idx_ in _filtered_processed_scenarios[scenario_id]['hypothesis_space']
                            [fd]['violations'].get(idx, []) if idx_ in data_indices])

    tuple_clean_score = math.exp(fd_prob*(compliance_num-violation_num))
    tuple_dirty_score = math.exp(fd_prob*(-compliance_num+violation_num))
    cond_p_clean = tuple_clean_score/(tuple_clean_score+tuple_dirty_score)

    return cond_p_clean


def get_average_cond_clean_prediction(indices, model, scenario_id):
    conditional_clean_probability_dict = dict()
    indices = set(indices)
    for idx in indices:
        cond_clean_prob = mean([compute_conditional_clean_prob(
            idx=idx, fd=fd, fd_prob=fd_prob, scenario_id=scenario_id,
            data_indices=indices) for fd, fd_prob in model.items()])  # whether to include the validation_indices or all the data_indices while computing the conditional clean probability
        conditional_clean_probability_dict[idx] = cond_clean_prob
    return conditional_clean_probability_dict

# %%


_model = dict(sorted(_models_dict[DATASET]['model'].items(), key=itemgetter(1),
                     reverse=True)[:10])
_clean_indices = set([idx for idx in _models_dict[DATASET]
                     ['predictions'] if _models_dict[DATASET]['predictions'][idx]])
_clean_probab_dict = get_average_cond_clean_prediction(
    _processed_df[DATASET].index, model=_model, scenario_id=DATASET)

# %%
# Check the clean label in the model file
for idx in _processed_df[DATASET].index:
    _clean = _clean_probab_dict[idx] >= 0.5
    if _clean == _models_dict[DATASET]['predictions'][idx]:
        continue
    print(idx, _clean, _models_dict[DATASET]['predictions'][idx])

# %%
# Check the aggreage model on the overall data
for hypothesis in _models_dict[DATASET]['model']:
    hypothesis_info_dict = _filtered_processed_scenarios[DATASET]['hypothesis_space'][hypothesis]

    support_pairs_num, violation_pairs_num = 0, 0
    for idx in hypothesis_info_dict['supports']:
        if idx not in _clean_indices:
            continue
        support_pairs_num += len(
            set(hypothesis_info_dict['supports'][idx]).intersection(_clean_indices))

    for idx in hypothesis_info_dict['violations']:
        if idx not in _clean_indices:
            continue
        violation_pairs_num += len(
            set(hypothesis_info_dict['violations'][idx]).intersection(_clean_indices))

    is_correct = round(support_pairs_num/(support_pairs_num+violation_pairs_num+1e-7),
                       3) == round(_models_dict[DATASET]['model'][hypothesis], 3)

    if not is_correct:
        print((support_pairs_num/(support_pairs_num+violation_pairs_num)),
              _models_dict[DATASET]['model'][hypothesis])

# %%
validation_indices = dict((dataset, sample(
    list(_models_dict[dataset]['predictions'].keys()), max(1, min(1000, len(_processed_df[dataset].index)-500)))) for dataset in _processed_df)

with open('./data/processed-data/validation_indices.json', 'w') as fp:
    json.dump(validation_indices, fp)

validation_indices = dict(
    (dataset, set(validation_indices[dataset])) for dataset in validation_indices)

with open('./data/processed-exp-data/validation_indices.pk', 'wb') as fp:
    pk.dump(validation_indices, fp)

# %%
print(1-np.mean([val for val in _models_dict[DATASET]['predictions'].values()]))
