import pandas as pd
import pickle
import json

data = pd.read_csv("data/ml-25m/ratings.csv")
movies = pd.read_csv("data/ml-25m/movies.csv")
movies = movies.set_index("movieId")

with open('config.json', 'r') as f:
    config_dict = json.load(f)

ROWS = int(config_dict["ROWS"])
SEED = int(config_dict["SEED"])
FREQUENCY = int(config_dict["FREQUENCY"])
USETIMESTAMPS = config_dict["USETIMESTAMPS"]
HIGHFREQUENCY = config_dict["HIGHFREQUENCY"]

if not(HIGHFREQUENCY):
    ## work with the first ROWS rows
    data = data.iloc[:ROWS]
else:
    # high frequency users
    high_frequency_users_dict = dict(filter(lambda x: x[1] >= FREQUENCY, dict(data['userId'].value_counts()).items()))
    high_frequency_users = list(high_frequency_users_dict.keys())
    data = data[data['userId'].isin(high_frequency_users)]

## remove rows where there are no genres listed
def filter_movies(movieId):
    movie = movies.loc[movieId]
    genres = movie["genres"].split("|")
    if "(no genres listed)" in genres:
        return False
    return True

data = data[data.apply(lambda row: filter_movies(row["movieId"]), axis=1)]

# randomly shuffle the rows (to shuffle the contexts), or USETIMESTAMPS to shuffle rows
if USETIMESTAMPS:
    data = data.sort_values(by=['timestamp']).reset_index(drop=True)
else:
    data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)

categories = set()

for i in range(len(data)):
    data_point = data.iloc[i]
    movie_id = int(data_point["movieId"])
    movie_categories = movies.loc[movie_id]["genres"].split('|')
    for category in movie_categories:
        categories.add(category)

# printing and saving the stats
stats_dict = {
    "NUM_CONTEXTS": len(data["userId"].unique()),
    "NUM_ARMS": len(list(categories)),
}

with open('stats.json', 'w') as f:
    json.dump(stats_dict, f)

# creating maps from user to index, and vice-versa
user_ids = sorted(list(data["userId"].unique()))
map_user_to_index = dict()
map_index_to_user = dict()
index = 0
for user_id in user_ids:
    map_user_to_index[user_id] = index
    map_index_to_user[index] = user_id
    index += 1

with open(config_dict["DATAPATH"], 'wb') as f:
    pickle.dump(data, f)

with open(config_dict["CATEGORIES_PATH"], 'wb') as f:
    pickle.dump(list(categories), f)

with open(config_dict["USER_TO_INDEX_PATH"], 'wb') as f:
    pickle.dump(map_user_to_index, f)

with open(config_dict["INDEX_TO_USER_PATH"], 'wb') as f:
    pickle.dump(map_index_to_user, f)
