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

## initialize policies
hedge = Hedge(NUM_CONTEXTS, NUM_ARMS, len(data))     # 2 users, 5 genres
parallelOPF = ParallelOPF(NUM_CONTEXTS, NUM_ARMS, ALPHA)

for i in range(len(data)):
    data_point = data.iloc[i]
    userId = int(data_point["userId"])
    movieId = int(data_point["userId"])

    hedge_recommended_genre = hedge.decision(userId - 1)    # context labels start from 0
    popf_recommended_genre = parallelOPF.decision(userId - 1)

    ## get rewards corresponding to the movie
    rewards = get_rewards(movieId)

    ## feedback rewards to the policies
    hedge.feedback(rewards)
    parallelOPF.feedback(rewards)

