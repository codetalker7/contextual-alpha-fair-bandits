import cvxpy as cp
import numpy as np
import pandas as pd
import pickle
import argparse
import random

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

# temporary constants
NUM_CONTEXTS = len(data["userId"].unique())
NUM_ARMS = len(categories)
ALPHA = float(args.ALPHA)
SMALL_REWARD = float(args.SMALL_REWARD)

def get_rewards(movieId):
    genres = movies.loc[movieId]["genres"].split("|")
    rewards = np.zeros((NUM_ARMS, ))

    for index in range(len(categories)):
        if categories[index] in genres:
            rewards[index] = 1
        else:
            rewards[index] = SMALL_REWARD

    return rewards

# sequence of contexts and reward vectors
context_sequence = list(data["userId"])
reward_sequence = []

for t in range(len(data)):
    data_point = data.iloc[t]
    movieId = int(data_point["movieId"])
    reward_sequence.append(get_rewards(movieId))
    

# need one vector variable for each constraint
variables = [cp.Variable(NUM_ARMS) for i in range(NUM_CONTEXTS)]

# constraints
constraints = []

## constraint for simplex
for i in range(NUM_CONTEXTS):
    constraints += [0 <= variables[i], variables[i] <= 1]
    constraints += [cp.sum(variables[i]) == 1]

objective_function = cp.expressions.constants.Constant(0)

for i in range(NUM_ARMS):
    cumulative_reward_arm = cp.expressions.constants.Constant(1)
    for t in range(len(data)):
        context_t = context_sequence[t]
        cumulative_reward_arm +=  reward_sequence[t][i] * variables[context_t - 1][i]
    cumulative_reward_arm = (cumulative_reward_arm ** ALPHA) / (1 - ALPHA)
    objective_function += cumulative_reward_arm

objective_function = -objective_function
obj = cp.Minimize(objective_function)

problem = cp.Problem(obj, constraints)
problem.solve()
