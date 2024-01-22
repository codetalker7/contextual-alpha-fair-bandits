import numpy as np
import pandas as pd
import pickle
import sys
import json
from tqdm import tqdm

## adding path for the policy classes
from os.path import dirname, abspath
class_path = dirname(dirname(abspath(__file__)))
sys.path.append(class_path)

## loading the config and stats
with open("config.json", "r") as f:
    config_dict = json.load(f)

with open("stats.json", "r") as f:
    stats_dict = json.load(f)

## loading the datasets
with open(config_dict["DATAPATH"], 'rb') as f:
    data = pickle.load(f)

with open(config_dict["CATEGORIES_PATH"], 'rb') as f:
    categories = list(pickle.load(f))

## load the user_to_index and index_to_user maps
with open(config_dict["USER_TO_INDEX_PATH"], 'rb') as f:
    map_user_to_index = pickle.load(f)

with open(config_dict["INDEX_TO_USER_PATH"], 'rb') as f:
    map_index_to_user = pickle.load(f)

## load the movie dataset
movies = pd.read_csv("data/ml-25m/movies.csv")
movies = movies.set_index("movieId")

## setting globals
NUM_CONTEXTS = int(stats_dict["NUM_CONTEXTS"])
NUM_ARMS = int(stats_dict["NUM_ARMS"])
SEED = int(config_dict["SEED"])
ALPHA = float(config_dict["ALPHA"])
SMALL_REWARD = float(config_dict["SMALLREWARD"])
APPROX_FACTOR = (1 - ALPHA) ** (-(1 - ALPHA)) 
if (config_dict["USETIMESTAMPS"] == 'True'):
    USETIMESTAMPS = True
else:
    USETIMESTAMPS = False

## offline optimal filename
OFFLINE_OPTIMAL_FILE = f"pickled_files/offline_optimal_alpha={ALPHA}_smallreward={SMALL_REWARD}_usetimestamps={USETIMESTAMPS}.pickle"

## random seed for numpy
np.random.seed(SEED)

## logging some stats
print("ARMS: ", NUM_ARMS)
print("CONTEXTS: ", NUM_CONTEXTS)
print("ALPHA: ", ALPHA)
print("APPROXIMATION FACTOR: ", APPROX_FACTOR)
print("SMALL_REWARD: ", SMALL_REWARD)
print("SEED: ", SEED)
print("USETIMESTAMPS:", USETIMESTAMPS)

# ## getting the offline optimal objectives
# with open("offline_optimal.pickle", "rb") as f:
#     offline_optimal_values = pickle.load(f)

## function to compute rewards for a movie
def get_rewards(movieId):
    genres = movies.loc[movieId]["genres"].split("|")
    rewards = np.zeros((NUM_ARMS, ))

    for index in range(len(categories)):
        if categories[index] in genres:
            rewards[index] = 1
        else:
            rewards[index] = SMALL_REWARD

    return rewards

## setting the reward function
reward_function = get_rewards
