import cvxpy as cp
import pandas as pd
import pickle
import json
import numpy as np
from tqdm import tqdm

def get_rewards(movieId):
    genres = movies.loc[movieId]["genres"].split("|")
    rewards = np.zeros((NUM_ARMS, ))

    for index in range(len(categories)):
        if categories[index] in genres:
            rewards[index] = 1
        else:
            rewards[index] = SMALL_REWARD

    return rewards

# load the config and stats
with open('config.json', 'r') as f:
    config_dict = json.load(f)

with open('stats.json', 'r') as f:
    stats_dict = json.load(f)

# load the data and categories
with open(config_dict["DATAPATH"], 'rb') as f:
    data = pickle.load(f)

with open(config_dict["CATEGORIES_PATH"],'rb') as f:
    categories = list(pickle.load(f))

# load the user_to_index and index_to_user maps
with open(config_dict["USER_TO_INDEX_PATH"], 'rb') as f:
    map_user_to_index = pickle.load(f)

with open(config_dict["INDEX_TO_USER_PATH"], 'rb') as f:
    map_index_to_user = pickle.load(f)

# load the movie dataset
movies = pd.read_csv("data/ml-25m/movies.csv")
movies = movies.set_index("movieId")

# temporary constants
NUM_CONTEXTS = int(stats_dict["NUM_CONTEXTS"])
NUM_ARMS = int(stats_dict["NUM_ARMS"])
ALPHA = float(config_dict["ALPHA"])
SMALL_REWARD = float(config_dict["SMALLREWARD"])
if (config_dict["USETIMESTAMPS"] == 'True'):
    USETIMESTAMPS = True
else:
    USETIMESTAMPS = False

# the reward function to use
reward_function = get_rewards
    
# need one vector variable for each constraint
variables = [cp.Variable(NUM_ARMS) for i in range(NUM_CONTEXTS)]

# constraints
constraints = []

## constraint for simplex
for i in range(NUM_CONTEXTS):
    constraints += [0 <= variables[i], variables[i] <= 1]
    constraints += [cp.sum(variables[i]) == 1]

cumulative_rewards = [cp.expressions.constants.Constant(1) for i in range(NUM_ARMS)]
offline_optimal_values = []

for t in tqdm(range(len(data))):
    data_point = data.iloc[t]
    movieId = int(data_point["movieId"])
    context_t = map_user_to_index[int(data_point["userId"])]
    rewards_t = reward_function(movieId)

    objective_function = cp.expressions.constants.Constant(0)

    for i in range(NUM_ARMS):
        cumulative_rewards[i] += rewards_t[i] * variables[context_t][i]
        objective_function += (cumulative_rewards[i] ** (1 - ALPHA)) / (1 - ALPHA)

    obj = cp.Minimize(-objective_function)
    problem = cp.Problem(obj, constraints)
    problem.solve()

    offline_optimal_values.append(-problem.value)

# saving the offline optimal values
filename = f'pickled_files/offline_optimal_alpha={ALPHA}_smallreward={SMALL_REWARD}_usetimestamps={USETIMESTAMPS}.pickle'
with open(filename, "wb") as f:
    pickle.dump(offline_optimal_values, f)
