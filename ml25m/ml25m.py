### preparing data
import numpy as np
import pandas as pd
import math
data = pd.read_csv("data/ml-25m/ratings.csv")
movies = pd.read_csv("data/ml-25m/movies.csv")
movies = movies.set_index("movieId")

# get data for only the first two users
data = data[data["userId"].isin([1, 2])]

# get filter movie categories, to be in Drama, action, comedy, adventure and crime
# movie_list = list(data["movieId"])
# def filter_function(ID):
#     genres = movies.loc[ID]["genres"].split("|")
#     genres_to_filter = ["Drama", "Action", "Comedy", "Adventure", "Crime"]
#
#     for genre in genres_to_filter:
#         if genre in genres:
#             return True
#
#     return False
# filtered_movie_list = list(filter(lambda ID : filter_function(ID), movie_list))

# get only those movies which belong to the filtered genres
# data = data[data["movieId"].isin(filtered_movie_list)]

# randomly shuffle the rows (to shuffle the contexts)
data = data.sample(frac=1).reset_index(drop=True)

########################

## running the policies
import sys
sys.path.append('/home/codetalker7/contextual-alpha-fair-bandits')
from Hedge import Hedge
from ParallelOPF import ParallelOPF

## 2 users, 5 genres
NUM_CONTEXTS = 2
NUM_ARMS = 5
ALPHA = 0.9

## associate genres to indices
genre_to_index = {
    "Drama": 0,
    "Action": 1,
    "Comedy": 2,
    "Adventure": 3,
    "Crime": 4
}

index_to_genre = {
    0: "Drama",
    1: "Action",
    2: "Comedy",
    3: "Adventure",
    4: "Crime"
}

## alpha-fair utility function
def utility(x):
    return math.pow(x, 1 - ALPHA) / (1 - ALPHA)

def get_rewards(movieId):
    genres = movies.loc[movieId]["genres"].split("|")
    rewards = np.zeros((NUM_ARMS, ))

    for index in list(index_to_genre.keys()):
        if index_to_genre[index] in genres:
            rewards[index] = 1
        else:
            rewards[index] = 0

    return rewards

def jains_fairness_index(v):
    n = v.shape[0]
    return ((v.sum())**2) / (n * (v * v).sum())

## initialize policies
hedge = Hedge(NUM_CONTEXTS, NUM_ARMS, len(data))     # 2 users, 5 genres
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
%matplotlib inline
import matplotlib.pyplot as plt

time = np.arange(1, len(data) + 1)

## plotting performance
hedge_performance = np.array(hedge_sum_rewards)[1:] * (1 / time)
popf_performance = np.array(popf_sum_rewards)[1:] * (1 / time)

plt.plot(time, hedge_performance, label="hedge")
plt.plot(time, popf_performance, label="parallel OPF")
plt.legend()
plt.show()

## plotting fairness
plt.plot(time, hedge_fairness_index, label="hedge")
plt.plot(time, popf_fairness_index, label="parallel OPF")
plt.legend()
plt.show()

###################
### Bandit Feedback setting
from ScaleFreeMAB import ScaleFreeMAB

scaleFreePolicy = ScaleFreeMAB(NUM_CONTEXTS, NUM_ARMS)

## keeping track of cumulative rewards
scaleFree_cum_rewards = [np.ones((NUM_ARMS, ))]

## jain's fairness index
scaleFree_fairness_index = []

## sum of rewards
scaleFree_sum_rewards = [0]

for t in range(len(data)):
    data_point = data.iloc[t]
    userId = int(data_point["userId"])
    movieId = int(data_point["movieId"])

    scaleFree_recommended_genre = scaleFreePolicy.decision(userId - 1)      # context labels start from 0

    ## characteristic vector for chosen arm
    scaleFree_char_vector = np.zeros((NUM_ARMS, ))
    scaleFree_char_vector[scaleFree_recommended_genre - 1] = 1

    ## get rewards corresponding to the movie
    rewards = get_rewards(movieId)

    ## update performance, only bandit feedback
    scaleFree_sum_rewards.append(scaleFree_sum_rewards[-1] + rewards[scaleFree_recommended_genre - 1])

    ## update cum rewards
    scaleFree_last_cum_rewards = scaleFree_cum_rewards[-1]

    scaleFree_cum_rewards.append(scaleFree_last_cum_rewards + rewards * scaleFree_char_vector)

    scaleFree_fairness_index.append(jains_fairness_index(scaleFree_cum_rewards[-1]))

    ## feedback rewards to the policies
    scaleFreePolicy.feedback(rewards[scaleFree_recommended_genre - 1])
