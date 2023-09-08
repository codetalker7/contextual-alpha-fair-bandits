import numpy as np
import pandas as pd
import pickle
import argparse
import sys
from tqdm import tqdm
sys.path.append('/home/codetalker7/contextual-alpha-fair-bandits')

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

## running the policies
from Hedge import Hedge
from ParallelOPF import ParallelOPF
from FairCB import FairCB
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

## getting the context distribution for fairCB
valueCounts = data["userId"].value_counts()
context_distribution = np.zeros((NUM_CONTEXTS, ))
for context_id in range(NUM_CONTEXTS):
    context_distribution[context_id] = valueCounts.loc[context_id + 1] / len(data)

## initialize policies
hedge = Hedge(NUM_CONTEXTS, NUM_ARMS, len(data))
parallelOPF = ParallelOPF(NUM_CONTEXTS, NUM_ARMS, ALPHA)
fairCB = FairCB(NUM_CONTEXTS, NUM_ARMS, 1 / (2 * NUM_ARMS), len(data), context_distribution)

## keeping track of cumulative rewards
hedge_cum_rewards = [np.ones((NUM_ARMS, ))]
popf_cum_rewards = [np.ones((NUM_ARMS, ))]
fairCB_cum_rewards = [np.ones((NUM_ARMS, ))]

## alpha-performance
hedge_alpha_performance = []
popf_alpha_performance = []
fairCB_alpha_performance = []

## jain's fairness index
hedge_fairness_index = []
popf_fairness_index = []
fairCB_fairness_index = []

## sum of rewards
hedge_sum_rewards = [0]
popf_sum_rewards = [0]
fairCB_sum_rewards = [0]

## standard regrets
hedge_standard_regret = []
popf_standard_regret = []
fairCB_standard_regret = []

## approximate regrets
hedge_approximate_regret = []
popf_approximate_regret = []
fairCB_approximate_regret = []

for t in tqdm(range(len(data))):
    data_point = data.iloc[t]
    userId = int(data_point["userId"])
    movieId = int(data_point["movieId"])

    hedge_recommended_genre = hedge.decision(userId - 1)    # context labels start from 0
    popf_recommended_genre = parallelOPF.decision(userId - 1)
    fairCB_recommended_genre = fairCB.decision(userId - 1)

    ## get rewards corresponding to the movie
    rewards = get_rewards(movieId)

    ## update performance
    hedge_sum_rewards.append(hedge_sum_rewards[-1] + rewards[hedge_recommended_genre - 1])
    popf_sum_rewards.append(popf_sum_rewards[-1] + rewards[popf_recommended_genre - 1])
    fairCB_sum_rewards.append(fairCB_sum_rewards[-1] + rewards[fairCB_recommended_genre - 1])

    ## update cum rewards
    hedge_last_cum_rewards = hedge_cum_rewards[-1]
    popf_last_cum_rewards = popf_cum_rewards[-1]
    fairCB_last_cum_rewards = fairCB_cum_rewards[-1]

    hedge_cum_rewards.append(hedge_last_cum_rewards + rewards * (hedge.weights / np.sum(hedge.weights)))
    popf_cum_rewards.append(popf_last_cum_rewards + rewards * parallelOPF.last_decision)
    fairCB_cum_rewards.append(fairCB_last_cum_rewards + rewards * fairCB.last_decision)

    ## updating alpha-performance
    hedge_alpha_performance.append((hedge_cum_rewards[-1] ** (1 - ALPHA) / (1 - ALPHA)).sum())
    popf_alpha_performance.append((popf_cum_rewards[-1] ** (1 - ALPHA) / (1 - ALPHA)).sum())
    fairCB_alpha_performance.append((fairCB_cum_rewards[-1] ** (1 - ALPHA) / (1 - ALPHA)).sum())

    ## update the fairness index
    hedge_fairness_index.append(jains_fairness_index(hedge_cum_rewards[-1]))
    popf_fairness_index.append(jains_fairness_index(popf_cum_rewards[-1]))
    fairCB_fairness_index.append(jains_fairness_index(fairCB_cum_rewards[-1]))

    ## update the standard regrets
    hedge_standard_regret.append(offline_optimal_values[t] - ((hedge_cum_rewards[-1] ** (1 - ALPHA)) / (1 - ALPHA)).sum())
    popf_standard_regret.append(offline_optimal_values[t] - ((popf_cum_rewards[-1] ** (1 - ALPHA)) / (1 - ALPHA)).sum())
    fairCB_standard_regret.append(offline_optimal_values[t] - ((fairCB_cum_rewards[-1] ** (1 - ALPHA)) / (1 - ALPHA)).sum())

    ## update the approximate regret
    hedge_approximate_regret.append(offline_optimal_values[t] - APPROX_FACTOR * ((hedge_cum_rewards[-1] ** (1 - ALPHA)) / (1 - ALPHA)).sum())
    popf_approximate_regret.append(offline_optimal_values[t] - APPROX_FACTOR * ((popf_cum_rewards[-1] ** (1 - ALPHA)) / (1 - ALPHA)).sum())
    fairCB_approximate_regret.append(offline_optimal_values[t] - APPROX_FACTOR * ((fairCB_cum_rewards[-1] ** (1 - ALPHA)) / (1 - ALPHA)).sum())

    ## feedback rewards to the policies
    hedge.feedback(rewards)
    parallelOPF.feedback(rewards)
    fairCB.feedback(rewards)

## plotting
# %matplotlib inline
import matplotlib.pyplot as plt

PERFORMANCE_PLOT_PATH = "performance_full_information.png"
ALPHA_PERFORMANCE_PLOT_PATH = "alpha_performance_full_information.png"
JAINS_FAIRNESS_PLOT_PATH = "jains_index_full_information.png"
APPROXIMATE_REGRET_PLOT_PATH = "approximate_regret_full_information.png"
STANDARD_REGRET_PLOT_PATH = "standard_regret_full_information.png"

time = np.arange(1, len(data) + 1)

## plotting performance
hedge_performance = np.array(hedge_sum_rewards)[1:] * (1 / time)
popf_performance = np.array(popf_sum_rewards)[1:] * (1 / time)
fairCB_performance = np.array(fairCB_sum_rewards)[1:] * (1 / time)

plt.figure(0)
plt.plot(time, hedge_performance, label="hedge")
plt.plot(time, popf_performance, label="parallel OPF")
plt.plot(time, fairCB_performance, label="fairCB")
plt.legend()
plt.savefig(PERFORMANCE_PLOT_PATH)

## plotting alpha-performance
plt.figure(1)
plt.plot(time, hedge_alpha_performance, label="hedge")
plt.plot(time, popf_alpha_performance, label="parallel OPF")
plt.plot(time, fairCB_alpha_performance, label="fairCB")
plt.legend()
plt.savefig(ALPHA_PERFORMANCE_PLOT_PATH)

## plotting fairness
plt.figure(2)
plt.plot(time, hedge_fairness_index, label="hedge")
plt.plot(time, popf_fairness_index, label="parallel OPF")
plt.plot(time, fairCB_fairness_index, label="fairCB")
plt.legend()
plt.savefig(JAINS_FAIRNESS_PLOT_PATH)

## plotting standard regrets
plt.figure(3)
plt.plot(time, hedge_standard_regret, label="hedge")
plt.plot(time, popf_standard_regret, label="parallel OPF")
plt.plot(time, fairCB_standard_regret, label="fairCB")
plt.legend()
plt.savefig(STANDARD_REGRET_PLOT_PATH)

## plotting approximate regrets
plt.figure(4)
plt.plot(time, hedge_approximate_regret, label="hedge")
plt.plot(time, popf_approximate_regret, label="parallel OPF")
plt.plot(time, fairCB_approximate_regret, label="fairCB")
plt.legend()
plt.savefig(APPROXIMATE_REGRET_PLOT_PATH)

