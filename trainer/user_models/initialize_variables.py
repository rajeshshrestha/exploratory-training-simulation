import json
from utils.logger import setup_logging
import logging
import pickle as pk

setup_logging("./logs", log_level='ERROR')

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
