from driver import *

from Hedge import Hedge
from ParallelOPF import ParallelOPF
from FairCB import FairCB
from utils import jains_fairness_index

import matplotlib.pyplot as plt

FOLDER_PREFIX = f''

## for varying nu experiments
FAIRNESS_VALUES_FILE = f'pickled_files/fairness_values_rows={config_dict["ROWS"]}_seed={SEED}_alpha={ALPHA}_smallreward={SMALL_REWARD}_usetimestamps={USETIMESTAMPS}_frequency={config_dict["FREQUENCY"]}_frequencymax={config_dict["FREQUENCY_MAX"]}_highfrequency={config_dict["HIGHFREQUENCY"]}_numnus={NUM_NUS}.pickle'

## getting the context distribution for fairCB
valueCounts = data["userId"].value_counts()
context_distribution = np.zeros((NUM_CONTEXTS, ))
for context_id in range(NUM_CONTEXTS):
    context_distribution[context_id] = valueCounts.loc[map_index_to_user[context_id]] / len(data)

## plot paths
JAINS_FAIRNESS_PLOT_PATH = "plots/jains_index_full_information.pdf"
APPROXIMATE_REGRET_PLOT_PATH = "plots/approximate_regret_full_information.pdf"
STANDARD_REGRET_PLOT_PATH = "plots/standard_regret_full_information.pdf"
VARYING_NUS_PLOT_PATH = "plots/jains_index_varying_nus.pdf"