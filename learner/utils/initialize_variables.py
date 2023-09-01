import json
from tqdm import tqdm
from utils.logger import setup_logging
import logging
import pandas as pd
import os
import pickle as pk
from copy import deepcopy

setup_logging("./logs", log_level="ERROR")

logger = logging.getLogger(__file__)

logger.info("Reading trainer_model.json file...")
with open('../data/processed-exp-data/trainer_model.json', 'r') as fp:
    models_dict = json.load(fp)

logger.info("Reading required_fds.pk file...")
with open('../data/processed-exp-data/required_fds.pk', 'rb') as fp:
    required_fds = pk.load(fp)


logger.info("Reading new_scenarios.json file...")
with open('../data/processed-exp-data/filtered_processed_scenarios.pk', 'rb') as fp:
    scenarios = pk.load(fp)

logger.info("Reading the processed datasets...")
with open('../data/processed-exp-data/processed_dfs.pk', 'rb') as fp:
    processed_dfs = pk.load(fp)

logger.info("Reading validation_indices...")
with open("../data/processed-exp-data/validation_indices.pk", 'rb') as fp:
    validation_indices_dict = pk.load(fp)
