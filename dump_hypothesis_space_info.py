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
parser.add_argument('--noise-ratio', default=None, type=float)

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

if args.noise_ratio and args.noise_ratio > 0:
    noisy_data_num = int(args.noise_ratio * len(data))
    noisy_data = []
    unique_vals = {field: list(data[field].unique()) for field in rel_cols}
    for i in range(noisy_data_num):
        noisy_data.append(
            {field: sample(unique_vals[field], k=1)[0] for field in rel_cols})
    noisy_df = pd.DataFrame(noisy_data)
    data = pd.concat([data, noisy_df], ignore_index=True)

data.to_csv(
    f"./data/preprocessed-data/{args.dataset_name}-clean-full.csv", index=False)

'''Load scenarios file'''
with open('scenarios.json', 'r') as fp:
    scenarios = json.load(fp)

'''Add new hypothesis space info in scenarios'''
scenarios[args.dataset_name] = {'hypothesis_space': fd_space}

'''Dump the info'''
with open('scenarios.json', 'w') as fp:
    json.dump(scenarios, fp)
