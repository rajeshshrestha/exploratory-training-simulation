import pandas as pd
import argparse
import os
from itertools import combinations
import json
from random import sample


'''Parse arguments'''
parser = argparse.ArgumentParser()
parser.add_argument('--drop-duplicates', default=False, action='store_true')
parser.add_argument('--data-path', required=True, type=str)
parser.add_argument('--dataset-name', required=True, type=str)
parser.add_argument('--rel-fields', type=str, nargs="+", default=None)
parser.add_argument('--injection-noise-ratio', default=None, type=float)

args = parser.parse_args()
print(args)

'''Check if the passed data file is present'''
assert os.path.isfile(
    args.data_path), f"File specified: {args.data_path} doesn't exist!!!"

'''Read data file'''
data = pd.read_csv(args.data_path)
data.rename(columns=dict((col, col.lower())
              for col in data.columns), inplace=True)
data.index = data.index.map(str)
data_cols = list(data.columns)

'''Relevant columns based on arguments passed'''
rel_cols = args.rel_fields if args.rel_fields is not None else data_cols

assert len(
    rel_cols) >= 2, f"Insufficient number of relevant columns: {rel_cols}"


'''Compute the fds'''
i, j = 1, 1
fd_space = []
for i in range(1, len(rel_cols)):
    for j in range(1, len(rel_cols)-i+1):
        for lhs in combinations(rel_cols, i):
            for rhs in combinations(set(rel_cols)-set(lhs), j):
                if i == 1:
                    lhs_val = lhs[0]
                else:
                    lhs_val = lhs
                fd_space.append(
                    {"cfd": "(" + ", ".join(lhs) + ")" + " => "+", ".join(rhs)})


if args.drop_duplicates:
    data = data[rel_cols].drop_duplicates()
else:
    data = data[rel_cols]

if args.injection_noise_ratio and args.injection_noise_ratio > 0:
    noisy_data_num = int(args.injection_noise_ratio * len(data))
    data_indices = list(data.index)
    for i in range(noisy_data_num):
        idx = sample(data_indices, k=1)[0]
        field = sample(rel_cols, k=1)[0]
        data.at[idx, field] = -1

os.makedirs("./data/preprocessed-data", exist_ok=True)
data.to_csv(
    f"./data/preprocessed-data/{args.dataset_name}-clean-full.csv")

'''Load scenarios file'''
if os.path.isfile('scenarios.json'):
    with open('scenarios.json', 'r') as fp:
        scenarios = json.load(fp)
else:
    scenarios = dict()

'''Add new hypothesis space info in scenarios'''
scenarios[args.dataset_name] = {'hypothesis_space': fd_space}

'''Dump the info'''
with open('scenarios.json', 'w') as fp:
    json.dump(scenarios, fp)
