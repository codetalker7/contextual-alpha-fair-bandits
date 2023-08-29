import numpy as np
from numpy.random import random
import pandas as pd
import argparse

from pandas.core.common import random_state

parser = argparse.ArgumentParser()
parser.add_argument('--ROWS', dest='ROWS', default=50000, help='Number of rows of the dataset to use.')
parser.add_argument('--SEED', dest='SEED', default=42, help='Random seed to shuffle the dataset.')
args = parser.parse_args()

data = pd.read_csv("data/ml-25m/ratings.csv")
movies = pd.read_csv("data/ml-25m/movies.csv")
movies = movies.set_index("movieId")

ROWS = int(args.ROWS)
SEED = int(args.SEED)
print("Number of rows to use: ", ROWS)
print("Random seed to shuffle dataset: ", SEED)

## work with the first ROWS rows
data = data.iloc[:ROWS]
data["movieId"].unique()

## remove rows where there are no genres listed
def filter_movies(movieId):
    movie = movies.loc[movieId]
    genres = movie["genres"].split("|")
    if "(no genres listed)" in genres:
        return False
    return True

data = data[data.apply(lambda row: filter_movies(row["movieId"]), axis=1)]

# randomly shuffle the rows (to shuffle the contexts)
data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)

categories = set()

for i in range(len(data)):
    data_point = data.iloc[i]
    movie_id = int(data_point["movieId"])
    movie_categories = movies.loc[movie_id]["genres"].split('|')
    for category in movie_categories:
        categories.add(category)

# saving the data to work with it later
import pickle

DATAPATH = "data.pickle"
CATEGORIES_PATH = "categories.pickle"

with open(DATAPATH, 'wb') as f:
    pickle.dump(data, f)

with open(CATEGORIES_PATH, 'wb') as f:
    pickle.dump(list(categories), f)


