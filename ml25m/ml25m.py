import numpy as np
import pandas as pd
import pickle
import argparse
import sys
sys.path.append('/home/codetalker7/contextual-alpha-fair-bandits')

parser = argparse.ArgumentParser()
parser.add_argument('--ALPHA', dest='ALPHA', default=0.5, help='Fairness level')
parser.add_argument('--SMALLREWARD', dest='SMALL_REWARD', default=0.001, help='Very small reward for the bad arm.')
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
from utils import jains_fairness_index

NUM_CONTEXTS = len(data["userId"].unique())
NUM_ARMS = len(categories)
ALPHA = float(args.ALPHA)
SMALL_REWARD = float(args.SMALL_REWARD)

print("ALPHA: ", ALPHA)
print("SMALL_REWARD: ", SMALL_REWARD)

def get_rewards(movieId):
    genres = movies.loc[movieId]["genres"].split("|")
    rewards = np.zeros((NUM_ARMS, ))

    for index in range(len(categories)):
        if categories[index] in genres:
            rewards[index] = 1
        else:
            rewards[index] = SMALL_REWARD

    return rewards

## initialize policies
hedge = Hedge(NUM_CONTEXTS, NUM_ARMS, len(data))
parallelOPF = ParallelOPF(NUM_CONTEXTS, NUM_ARMS, ALPHA)

## keeping track of cumulative rewards
hedge_cum_rewards = [np.ones((NUM_ARMS, ))]
popf_cum_rewards = [np.ones((NUM_ARMS, ))]

## jain's fairness index
hedge_fairness_index = []
popf_fairness_index = []

## sum of rewards
hedge_sum_rewards = [0]
popf_sum_rewards = [0]

for t in range(len(data)):
    data_point = data.iloc[t]
    userId = int(data_point["userId"])
    movieId = int(data_point["movieId"])

    hedge_recommended_genre = hedge.decision(userId - 1)    # context labels start from 0
    popf_recommended_genre = parallelOPF.decision(userId - 1)

    ## get rewards corresponding to the movie
    rewards = get_rewards(movieId)

    ## update performance
    hedge_sum_rewards.append(hedge_sum_rewards[-1] + rewards[hedge_recommended_genre - 1])
    popf_sum_rewards.append(popf_sum_rewards[-1] + rewards[popf_recommended_genre - 1])

    ## update cum rewards
    hedge_last_cum_rewards = hedge_cum_rewards[-1]
    popf_last_cum_rewards = popf_cum_rewards[-1]

    hedge_cum_rewards.append(hedge_last_cum_rewards + rewards * (hedge.weights / np.sum(hedge.weights)))
    popf_cum_rewards.append(popf_last_cum_rewards + rewards * parallelOPF.last_decision)

    hedge_fairness_index.append(jains_fairness_index(hedge_cum_rewards[-1]))
    popf_fairness_index.append(jains_fairness_index(popf_cum_rewards[-1]))

    ## feedback rewards to the policies
    hedge.feedback(rewards)
    parallelOPF.feedback(rewards)

## plotting
# %matplotlib inline
import matplotlib.pyplot as plt

PERFORMANCE_PLOT_PATH = "performance_full_information.png"
JAINS_FAIRNESS_PLOT_PATH = "jains_index_full_information.png"

time = np.arange(1, len(data) + 1)

## plotting performance
hedge_performance = np.array(hedge_sum_rewards)[1:] * (1 / time)
popf_performance = np.array(popf_sum_rewards)[1:] * (1 / time)

plt.figure(0)
plt.plot(time, hedge_performance, label="hedge")
plt.plot(time, popf_performance, label="parallel OPF")
plt.legend()
plt.savefig(PERFORMANCE_PLOT_PATH)

## plotting fairness
plt.figure(1)
plt.plot(time, hedge_fairness_index, label="hedge")
plt.plot(time, popf_fairness_index, label="parallel OPF")
plt.legend()
plt.savefig(JAINS_FAIRNESS_PLOT_PATH)
