import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--SEED', dest='SEED', default=42, help='Random seed to shuffle the dataset.')
parser.add_argument('--FREQUENCY', dest='FREQUENCY', default=5000, help='Minimum frequency of a context in the resultant dataset.')
parser.add_argument('--USETIMESTAMPS', dest='USETIMESTAMPS', default=False, help='Boolean determining whether the timestamps given in the dataset will be used to shuffle the rows.')
args = parser.parse_args()

data = pd.read_csv("data/ml-25m/ratings.csv")
movies = pd.read_csv("data/ml-25m/movies.csv")
movies = movies.set_index("movieId")

SEED = int(args.SEED)
FREQUENCY = int(args.FREQUENCY)
USETIMESTAMPS = bool(args.USETIMESTAMPS)

print("Use timestamps to shuffle rows?: ", USETIMESTAMPS)
print("Random seed to shuffle dataset (if USETIMESTAMPS is false): ", SEED)
print("Minimum Frequency Required: ", FREQUENCY)

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

# printing the stats
print("Number of contexts: ", len(data["userId"].unique()))
print("Number of arms:", len(list(categories)))

# saving the data to work with it later
import pickle

DATAPATH = "data.pickle"
CATEGORIES_PATH = "categories.pickle"

with open(DATAPATH, 'wb') as f:
    pickle.dump(data, f)

with open(CATEGORIES_PATH, 'wb') as f:
    pickle.dump(list(categories), f)

