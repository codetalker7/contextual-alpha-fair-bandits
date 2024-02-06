from driver import *

from ScaleFreeMAB import ScaleFreeMAB
from ParallelScaleFreeMAB import ParallelScaleFreeMAB
from utils import jains_fairness_index
import matplotlib.pyplot as plt

## bandit information experiments
BANDIT_INFORMATION_FILE = f'pickled_files/bandit_information_rows={ROWS}_seed={SEED}_alpha={ALPHA}_smallreward={SMALL_REWARD}_usetimestamps={USETIMESTAMPS}_frequency={FREQUENCY}_frequencymax={FREQUENCY_MAX}_faircbfairness={FAIRCBFAIRNESS}_highfrequency={HIGHFREQUENCY}.pickle'

JAINS_FAIRNESS_PLOT_PATH = "plots/jains_index_bandit_information.pdf"
STANDARD_REGRET_PLOT_PATH = "plots/standard_regret_bandit_information.pdf"
APPROXIMATE_REGRET_PLOT_PATH = "plots/approximate_regret_bandit_information.pdf"