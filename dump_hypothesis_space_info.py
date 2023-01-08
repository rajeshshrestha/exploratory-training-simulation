import pandas as pd
import argparse
import os
from itertools import combinations
import json


'''Parse arguments'''
parser = argparse.ArgumentParser()
parser.add_argument('--drop-duplicates', default=False, action='store_true')
parser.add_argument('--data-path', required=True, type=str)
parser.add_argument('--dataset-name', required=True, type=str)
parser.add_argument('--rel-fields', type=str, nargs="+", default=None)

args = parser.parse_args()
print(args)

'''Check if the passed data file is present'''
assert os.path.isfile(args.data_path), f"File specified: {args.data_path} doesn't exist!!!"

'''Read data file'''
data = pd.read_csv(args.data_path)
data_cols = list(data.columns)

'''Relevant columns based on arguments passed'''
rel_cols = [field for field in args.rel_fields if field in data_cols] if args.rel_fields is not None else data_cols

assert len(rel_cols) >=2, f"Insufficient number of relevant columns: {rel_cols}"


'''Compute the fds'''
i,j=1,1
fd_space = []
for i in range(1,len(rel_cols)):
    for j in range(1, len(rel_cols)-i+1):
        for lhs in combinations(rel_cols, i):
            for rhs in combinations(set(rel_cols)-set(lhs), j):
                if i == 1:
                    lhs_val = lhs[0]
                else:
                    lhs_val = lhs
                fd_space.append({"cfd": "(" + ", ".join(lhs) +  ")" + " => "+", ".join(rhs)})


if args.drop_duplicates:
    data[rel_cols].drop_duplicates().to_csv(f"./data/preprocessed-data/{args.dataset_name}-clean-full.csv", index=False)
else:
    data[rel_cols].to_csv(f"./data/preprocessed-data/{args.dataset_name}-clean-full.csv", index=False)


'''Load scenarios file'''
with open('scenarios.json', 'r') as fp:
    scenarios = json.load(fp)

'''Add new hypothesis space info in scenarios'''
scenarios[args.dataset_name] = fd_space

'''Dump the info'''
with open('scenarios.json', 'w') as fp:
    json.dump(scenarios, fp)