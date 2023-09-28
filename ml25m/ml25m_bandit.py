import numpy as np
import pandas as pd
import pickle
import argparse
import sys
from tqdm import tqdm

## adding path for the policy classes
from os.path import dirname, abspath
class_path = dirname(dirname(abspath(__file__)))
sys.path.append(class_path)

parser = argparse.ArgumentParser()
parser.add_argument('--ALPHA', dest='ALPHA', default=0.5, help='Fairness level')
parser.add_argument('--SMALLREWARD', dest='SMALL_REWARD', default=0.001, help='Very small reward for the bad arm.')
parser.add_argument('--SEED', dest='SEED', default=42, help='Random seed to have reproducible results.')
args = parser.parse_args()

with open("data.pickle", 'rb') as f:
    data = pickle.load(f)

with open("categories.pickle", 'rb') as f:
    categories = list(pickle.load(f))

movies = pd.read_csv("data/ml-25m/movies.csv")
movies = movies.set_index("movieId")

from ScaleFreeMAB import ScaleFreeMAB
from ParallelScaleFreeMAB import ParallelScaleFreeMAB
from utils import jains_fairness_index

NUM_CONTEXTS = len(data["userId"].unique())
NUM_ARMS = len(categories)
ALPHA = float(args.ALPHA)
SMALL_REWARD = float(args.SMALL_REWARD)
APPROX_FACTOR = (1 - ALPHA) ** (-(1 - ALPHA)) 

## random seed for numpy
np.random.seed(int(args.SEED))

print("ARMS: ", NUM_ARMS)
print("CONTEXTS: ", NUM_CONTEXTS)
print("ALPHA: ", ALPHA)
print("APPROXIMATION FACTOR: ", APPROX_FACTOR)
print("SMALL_REWARD: ", SMALL_REWARD)
print("SEED: ", int(args.SEED))

# getting the offline optimal objectives
with open("offline_optimal.pickle", "rb") as f:
    offline_optimal_values = pickle.load(f)

def get_rewards(movieId):
    genres = movies.loc[movieId]["genres"].split("|")
    rewards = np.zeros((NUM_ARMS, ))

    for index in range(len(categories)):
        if categories[index] in genres:
            rewards[index] = 1
        else:
            rewards[index] = SMALL_REWARD

    return rewards

scaleFreePolicy = ScaleFreeMAB(NUM_CONTEXTS, NUM_ARMS)
parallelScaleFreePolicy = ParallelScaleFreeMAB(NUM_CONTEXTS, NUM_ARMS, ALPHA)

## keeping track of cumulative rewards
scaleFree_cum_rewards = [np.ones((NUM_ARMS, ))]
parallelScaleFree_cum_rewards = [np.ones((NUM_ARMS, ))]

## alpha-performance
scaleFree_alpha_performance = []
parallelScaleFree_alpha_performance = []

## jain's fairness index
scaleFree_fairness_index = []
parallelScaleFree_fairness_index = []

## sum of rewards
scaleFree_sum_rewards = [0]
parallelScaleFree_sum_rewards = [0]

## approximate regrets
scaleFree_approximate_regret = []
parallelScaleFree_approximate_regret = []

for t in tqdm(range(len(data))):
    data_point = data.iloc[t]
    userId = int(data_point["userId"])
    movieId = int(data_point["movieId"])

    scaleFree_recommended_genre = scaleFreePolicy.decision(userId - 1)      # context labels start from 0
    parallelScaleFree_recommended_genre = parallelScaleFreePolicy.decision(userId - 1)

    ## characteristic vector for chosen arm
    scaleFree_char_vector = np.zeros((NUM_ARMS, ))
    scaleFree_char_vector[scaleFree_recommended_genre - 1] = 1

    parallelScaleFree_char_vector = np.zeros((NUM_ARMS, ))
    parallelScaleFree_char_vector[parallelScaleFree_recommended_genre - 1] = 1

    ## get rewards corresponding to the movie
    rewards = get_rewards(movieId)

    ## update performance
    scaleFree_sum_rewards.append(scaleFree_sum_rewards[-1] + rewards[scaleFree_recommended_genre - 1])
    parallelScaleFree_sum_rewards.append(parallelScaleFree_sum_rewards[-1] + rewards[parallelScaleFree_recommended_genre - 1])

    ## update cum rewards
    scaleFree_last_cum_rewards = scaleFree_cum_rewards[-1]
    parallelScaleFree_last_cum_rewards = parallelScaleFree_cum_rewards[-1]

    scaleFree_cum_rewards.append(scaleFree_last_cum_rewards + rewards * scaleFree_char_vector)
    parallelScaleFree_cum_rewards.append(parallelScaleFree_last_cum_rewards + rewards * parallelScaleFree_char_vector)

    ## updating alpha-performance
    scaleFree_alpha_performance.append((scaleFree_cum_rewards[-1] ** (1 - ALPHA) / (1 - ALPHA)).sum()) 
    parallelScaleFree_alpha_performance.append((parallelScaleFree_cum_rewards[-1] ** (1 - ALPHA) / (1 - ALPHA)).sum())

    ## update the fairness index
    scaleFree_fairness_index.append(jains_fairness_index(scaleFree_cum_rewards[-1]))
    parallelScaleFree_fairness_index.append(jains_fairness_index(parallelScaleFree_cum_rewards[-1]))

    ## update the approximate regrets
    scaleFree_approximate_regret.append(offline_optimal_values[t] - APPROX_FACTOR * ((scaleFree_cum_rewards[-1] ** (1 - ALPHA)) / (1 - ALPHA)).sum())
    parallelScaleFree_approximate_regret.append(offline_optimal_values[t] - APPROX_FACTOR * ((parallelScaleFree_cum_rewards[-1] ** (1 - ALPHA)) / (1 - ALPHA)).sum())

    ## feedback rewards to the policies
    scaleFreePolicy.feedback(rewards[scaleFree_recommended_genre - 1])
    parallelScaleFreePolicy.feedback(rewards[parallelScaleFree_recommended_genre - 1])

## plotting
# %matplotlib inline
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams["figure.figsize"] = (5, 4)

PERFORMANCE_PLOT_PATH = "plots/performance_bandit_information.pdf"
ALPHA_PERFORMANCE_PLOT_PATH = "plots/alpha_performance_bandit_information.pdf"
JAINS_FAIRNESS_PLOT_PATH = "plots/jains_index_bandit_information.pdf"
APPROXIMATE_REGRET_PLOT_PATH = "plots/approximate_regret_bandit_information.pdf"

time = np.arange(1, len(data) + 1)

## plotting performance
scaleFree_performance = np.array(scaleFree_sum_rewards)[1:] * (1 / time)
parallelScaleFree_performance = np.array(parallelScaleFree_sum_rewards)[1:] * (1 / time)

plt.figure(0)
plt.plot(time, scaleFree_performance, label="Putta \& Aggarwal, 2022")
plt.plot(time, parallelScaleFree_performance, label=r"$\alpha\textsc{-FairCB}$")
plt.legend(loc="upper left", fontsize="large")
plt.xlabel("Time", fontsize="large")
plt.ylabel("Performance", fontsize="large")
plt.savefig(PERFORMANCE_PLOT_PATH)

## plotting alpha-performance
plt.figure(1)
plt.plot(time, scaleFree_alpha_performance, label="Putta \& Aggarwal, 2022")
plt.plot(time, parallelScaleFree_alpha_performance, label=r"$\alpha\textsc{-FairCB}$")
plt.legend(loc="upper left", fontsize="large")
plt.xlabel("Time", fontsize="large")
plt.ylabel(r'$\alpha$-Performance', fontsize="large")
plt.savefig(ALPHA_PERFORMANCE_PLOT_PATH, bbox_inches='tight', pad_inches=0.01)

## plotting fairness
plt.figure(2)
plt.plot(time, scaleFree_fairness_index, label="Putta \& Aggarwal, 2022")
plt.plot(time, parallelScaleFree_fairness_index, label=r"$\alpha\textsc{-FairCB}$")
plt.legend(loc="center right", fontsize="large")
plt.xlabel("Time", fontsize="large")
plt.ylabel("Jain's Fairness Index", fontsize="large")
plt.savefig(JAINS_FAIRNESS_PLOT_PATH, bbox_inches='tight', pad_inches=0.01)

## plotting regrets
plt.figure(3)
plt.plot(time, scaleFree_approximate_regret, label="Putta \& Aggarwal, 2022")
plt.plot(time, parallelScaleFree_approximate_regret, label=r"$\alpha\textsc{-FairCB}$")
plt.legend(loc="center right", fontsize="large")
plt.xlabel("Time", fontsize="large")
plt.ylabel("Approximate Regret", fontsize="large")
plt.savefig(APPROXIMATE_REGRET_PLOT_PATH, bbox_inches='tight', pad_inches=0.01)
