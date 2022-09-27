# %% [markdown]
# ## Import libraries

# %%
import pandas as pd
import json
from tqdm import tqdm
from multiprocessing import Pool
import os

# %% [markdown]
# ## Read raw data

# %%
data_path = "./data/raw-data/omdb-clean-full.csv"

raw_df = pd.read_csv(data_path)
raw_df.rename(columns=dict((col, col.lower()) for col in raw_df.columns), inplace=True)
raw_df.head()


# %%
## Read Scenario3 info for the functional dependencies

with open("./scenarios.json", 'r') as fp:
    scenarios = json.load(fp)
required_scenario_info = scenarios["3"]
hypothesis_space = [hypothesis['cfd'] for hypothesis in required_scenario_info['hypothesis_space']]
print(hypothesis_space)

# %% [markdown]
# ## Add info to new scenarios dict and dump the file

# %%
def parse_hypothesis(fd):
    lfd, rfd = fd.split("=>")

    '''Parse left fd and separate out the attributes'''
    left_attributes = lfd.strip().strip("(").strip(")").split(",")
    right_attributes = rfd.strip("(").strip(")").split(",")

    left_attributes = [attribute.strip() for attribute in left_attributes]
    right_attributes = [attribute.strip() for attribute in right_attributes]

    return left_attributes, right_attributes

# %%
def is_support_violation(fd_components, tuple_1, tuple_2):
    '''Parse the hypothesis'''
    lfd, rfd = fd_components

    '''Violation check is only needed if the lfd values are same in both tuples otherwise it's not a violation'''
    is_left_same = all(tuple_1[left_attribute] == tuple_2[left_attribute] for left_attribute in lfd)

    if is_left_same:
        is_right_same = all(tuple_1[left_attribute] == tuple_2[left_attribute] for left_attribute in rfd)
        if is_right_same:
            return True, False
        else:
            return False, True
    else:
        return False, False


# %%
def get_support_violation_tuples(data, idx, fd_components):
    supports = []
    violations = []
    for idx_ in data.index:
        is_support, is_violation = is_support_violation(fd_components=fd_components, tuple_1=data.iloc[idx], tuple_2=data.iloc[idx_])
        if is_support:
            supports.append(idx_)
        elif is_violation:
            violations.append(idx_)
    return supports, violations
        

# %%
def get_hypothesis_info_dict(hypothesis):
    '''Extract left and right attributes from the hypothesis as list of attributes'''
    lfd, rfd = parse_hypothesis(hypothesis)
    info_dict = {'lfd': lfd, 'rfd':rfd}

    '''Find pairwise violations of each tuple with respect to other tuples in the dataset'''
    info_dict['supports'] = dict()
    info_dict['violations'] = dict()
    for idx in tqdm(raw_df.index):
        supports, violations = get_support_violation_tuples(data=raw_df[lfd+rfd], idx=idx, fd_components=(lfd, rfd))
        info_dict['supports'][idx] = supports
        info_dict['violations'][idx] = violations
    
    return info_dict

# get_hypothesis_info_dict(hypothesis_space[0])
# hypothesis_space[0]

# %%
cpu_num = os.cpu_count()
# cpu_num = 2

# %%
new_scenarios_dict = dict()
new_scenarios_dict['omdb'] = dict()
new_scenarios_dict['omdb']['hypothesis_space'] = dict()


with Pool(cpu_num) as p:
    hypothesis_info = p.map(get_hypothesis_info_dict, hypothesis_space)

for hypothesis, info_dict in zip(hypothesis_space, hypothesis_info):
    new_scenarios_dict['omdb']['hypothesis_space'][hypothesis] = info_dict

# for hypothesis in tqdm(hypothesis_space):
    
    # '''Extract left and right attributes from the hypothesis as list of attributes'''
    # lfd, rfd = parse_hypothesis(hypothesis)
    # new_scenarios_dict['omdb']['hypothesis_space'][hypothesis] = {'lfd': lfd, 'rfd':rfd}

    # '''Find pairwise violations of each tuple with respect to other tuples in the dataset'''
    # new_scenarios_dict['omdb']['hypothesis_space'][hypothesis]['violations'] = dict()
    # for idx in raw_df.index:
    #     new_scenarios_dict['omdb']['hypothesis_space'][hypothesis]['violations'][idx]=get_violation_tuples(data=raw_df, idx=idx, fd=hypothesis)

# %%
with open("./new_scenarios.json", 'w') as fp:
    json.dump(new_scenarios_dict, fp)

# %%
with open("./new_scenarios 2.json", 'r') as fp:
    new_scenario = json.load(fp) 

# %%
new_scenario[hypothesis[0]]

# %%



