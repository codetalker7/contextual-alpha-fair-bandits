import numpy as np
import pandas as pd
import math
import pickle

with open("data.pickle", 'rb') as f:
    data = pickle.load(f)

with open("categories.pickle", 'rb') as f:
    categories = list(pickle.load(f))

movies = pd.read_csv("data/ml-25m/movies.csv")
movies = movies.set_index("movieId")

# randomly shuffle the rows (to shuffle the contexts)
# data = data.sample(frac=1).reset_index(drop=True)

import sys
sys.path.append('/home/codetalker7/contextual-alpha-fair-bandits')
from ScaleFreeMAB import ScaleFreeMAB
from ParallelScaleFreeMAB import ParallelScaleFreeMAB

## 2 users, 5 genres
NUM_CONTEXTS = len(data["userId"].unique())
NUM_ARMS = len(categories)
ALPHA = 0.9

## alpha-fair utility function
def utility(x):
    return math.pow(x, 1 - ALPHA) / (1 - ALPHA)

def get_rewards(movieId):
    genres = movies.loc[movieId]["genres"].split("|")
    rewards = np.zeros((NUM_ARMS, ))

    for index in range(len(categories)):
        if categories[index] in genres:
            rewards[index] = 1
        else:
            rewards[index] = 0

    return rewards

def jains_fairness_index(v):
    n = v.shape[0]
    return ((v.sum())**2) / (n * (v * v).sum())

scaleFreePolicy = ScaleFreeMAB(NUM_CONTEXTS, NUM_ARMS)
parallelScaleFreePolicy = ParallelScaleFreeMAB(NUM_CONTEXTS, NUM_ARMS, ALPHA)

## keeping track of cumulative rewards
scaleFree_cum_rewards = [np.ones((NUM_ARMS, ))]
parallelScaleFree_cum_rewards = [np.ones((NUM_ARMS, ))]

## jain's fairness index
scaleFree_fairness_index = []
parallelScaleFree_fairness_index = []

## sum of rewards
scaleFree_sum_rewards = [0]
parallelScaleFree_sum_rewards = [0]

for t in range(len(data)):
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

    scaleFree_fairness_index.append(jains_fairness_index(scaleFree_cum_rewards[-1]))
    parallelScaleFree_fairness_index.append(jains_fairness_index(parallelScaleFree_cum_rewards[-1]))

    ## feedback rewards to the policies
    scaleFreePolicy.feedback(rewards[scaleFree_recommended_genre - 1])
    parallelScaleFreePolicy.feedback(rewards[parallelScaleFree_recommended_genre - 1])

## plotting
%matplotlib inline
import matplotlib.pyplot as plt

time = np.arange(1, len(data) + 1)

## plotting performance
scaleFree_performance = np.array(scaleFree_sum_rewards)[1:] * (1 / time)
parallelScaleFree_performance = np.array(parallelScaleFree_sum_rewards)[1:] * (1 / time)

plt.plot(time, scaleFree_performance, label="scaleFree")
plt.plot(time, parallelScaleFree_performance, label="parallelScaleFree")
plt.legend()
plt.show()

## plotting fairness
plt.plot(time, scaleFree_fairness_index, label="scaleFree")
plt.plot(time, parallelScaleFree_fairness_index, label="parallelScaleFree")
plt.legend()
plt.show()