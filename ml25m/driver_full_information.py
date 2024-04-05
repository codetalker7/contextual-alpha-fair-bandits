from driver import *

from Hedge import Hedge
from ParallelOPF import ParallelOPF
from FairCB import FairCB
from IndependentOPF import IndependentOPF
from utils import jains_fairness_index
import matplotlib.pyplot as plt

## full information experiments
FULL_INFORMATION_FILE = f'pickled_files/full_information_rows={ROWS}_seed={SEED}_alpha={ALPHA}_smallreward={SMALL_REWARD}_usetimestamps={USETIMESTAMPS}_frequency={FREQUENCY}_frequencymax={FREQUENCY_MAX}_faircbfairness={FAIRCBFAIRNESS}_highfrequency={HIGHFREQUENCY}.pickle'

## for varying nu experiments
FAIRNESS_VALUES_FILE = f'pickled_files/fairness_values_rows={ROWS}_seed={SEED}_alpha={ALPHA}_smallreward={SMALL_REWARD}_usetimestamps={USETIMESTAMPS}_frequency={FREQUENCY}_frequencymax={FREQUENCY_MAX}_highfrequency={HIGHFREQUENCY}_numnus={NUM_NUS}_varyingnurounds={VARYING_NU_ROUNDS}.pickle'

## for varying alpha experiments
CUMULATIVE_REWARDS_FILE = f'pickled_files/cumulative_rewards_rows={ROWS}_seed={SEED}_smallreward={SMALL_REWARD}_usetimestamps={USETIMESTAMPS}_frequency={FREQUENCY}_frequencymax={FREQUENCY_MAX}_highfrequency={HIGHFREQUENCY}_numalphas={NUM_ALPHAS}.pickle'

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
VARYING_ALPHAS_PLOT_PATH = "plots/jains_index_varying_alphas.pdf"
AVERAGED_CUMULATIVE_REWARDS_PATH = "plots/avg_cumulative_rewards.pdf"
