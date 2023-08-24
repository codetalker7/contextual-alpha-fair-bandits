import numpy as np
import pandas as pd
import math
data = pd.read_csv("data/ml-25m/ratings.csv")
movies = pd.read_csv("data/ml-25m/movies.csv")
movies = movies.set_index("movieId")

ROWS = 10000
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


