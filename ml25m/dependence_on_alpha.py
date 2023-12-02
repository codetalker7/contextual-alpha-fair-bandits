import numpy as np
import pandas as pd
import argparse
import sys
import pickle
from tqdm import tqdm

## adding path for the policy classes
from os.path import dirname, abspath
class_path = dirname(dirname(abspath(__file__)))
sys.path.append(class_path)

parser = argparse.ArgumentParser()
parser.add_argument('--SMALLREWARD', dest='SMALL_REWARD', default=0.001, help='Very small reward for the bad arm.')
parser.add_argument('--SEED', dest='SEED', default=42, help='Random seed to have reproducible results.')
args = parser.parse_args()

with open("data.pickle", 'rb') as f:
    data = pickle.load(f)

with open("categories.pickle", 'rb') as f:
    categories = list(pickle.load(f))

movies = pd.read_csv("data/ml-25m/movies.csv")
movies = movies.set_index("movieId")

NUM_CONTEXTS = len(data["userId"].unique())
NUM_ARMS = len(categories)
alphas = np.linspace(0, 1, 101).tolist()[:-1]
SMALL_REWARD = float(args.SMALL_REWARD)

## random seed for numpy
np.random.seed(int(args.SEED))

print("ARMS: ", NUM_ARMS)
print("CONTEXTS: ", NUM_CONTEXTS)
print("SMALL_REWARD: ", SMALL_REWARD)
print("SEED: ", int(args.SEED))

def get_rewards(movieId):
    genres = movies.loc[movieId]["genres"].split("|")
    rewards = np.zeros((NUM_ARMS, ))

    for index in range(len(categories)):
        if categories[index] in genres:
            rewards[index] = 1
        else:
            rewards[index] = SMALL_REWARD

    return rewards

## map userId's to an index in the range [0, NUM_CONTEXTS - 1]
user_ids = sorted(list(data["userId"].unique()))
map_user_to_index = dict()
index = 0
for user_id in user_ids:
    map_user_to_index[user_id] = index
    index += 1

## getting the context distribution for fairCB
valueCounts = data["userId"].value_counts()
context_distribution = np.zeros((NUM_CONTEXTS, ))
for user_id in user_ids:
    context_distribution[map_user_to_index[user_id]] = valueCounts.loc[user_id] / len(data)

## running the policies
from ParallelOPF import ParallelOPF
from FairCB import FairCB
from utils import jains_fairness_index

policies = [ParallelOPF(NUM_CONTEXTS, NUM_ARMS, alphas[i]) for i in range(len(alphas))]
faircb_policies = [FairCB(NUM_CONTEXTS, NUM_ARMS, alphas[i] / len(data), len(data), context_distribution) for i in range(len(alphas))] 

cumulative_rewards = [[np.ones(NUM_ARMS, )] for i in range(len(alphas))]
faircb_cumulative_rewards = [[np.ones(NUM_ARMS, )] for i in range(len(alphas))]

for t in tqdm(range(len(data))):
    data_point = data.iloc[t]
    userId = int(data_point["userId"])
    movieId = int(data_point["movieId"])

    # map user id to an index in the range [0, NUM_CONTEXTS - 1]
    userId = map_user_to_index[userId]

    recommended_genres = [policies[i].decision(userId - 1) for i in range(len(alphas))]
    faircb_recommended_genres = [faircb_policies[i].decision(userId - 1) for i in range(len(alphas))]

    ## get rewards corresponding to the movie
    rewards = get_rewards(movieId)

    ## update cumulative rewards
    for i in range(len(alphas)):
        last_cum_rewards = cumulative_rewards[i][-1]
        cumulative_rewards[i].append(last_cum_rewards + rewards * policies[i].last_decision)

        faircb_last_cum_rewards = faircb_cumulative_rewards[i][-1]
        faircb_cumulative_rewards[i].append(faircb_last_cum_rewards + rewards * faircb_policies[i].last_decision)

    ## feedback rewards to the policies
    for i in range(len(alphas)):
        policies[i].feedback(rewards)
        faircb_policies[i].feedback(rewards)

## plotting
# %matplotlib inline
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams["figure.figsize"] = (5, 4)

AVERAGED_CUMULATIVE_REWARDS_PATH = "plots/avg_cumulative_rewards.pdf"
ALPHA_FAIR_OBJECTIVE_VARIATION_PATH = "plots/alpha_objective_variation.pdf"
FAIRNESS_VARIATION_PATH = "plots/fairness_variation.pdf" 

plt.figure(0)
plt.plot(alphas, [cumulative_rewards[i][-1].sum() / NUM_ARMS for i in range(len(alphas))], label=r"$\alpha$-\textsc{FairCB}")
plt.plot(alphas, [faircb_cumulative_rewards[i][-1].sum() / NUM_ARMS for i in range(len(alphas))], label=r"\textsc{FairCB}")
plt.legend(loc="upper left", fontsize="large")
plt.xlabel(r"$\alpha$", fontsize="large")
plt.ylabel("Average Cumulative Rewards", fontsize="large")
plt.savefig(AVERAGED_CUMULATIVE_REWARDS_PATH)

plt.figure(1)
plt.plot(alphas, [(cumulative_rewards[i][-1] ** (1 - alphas[i]) / (1 - alphas[i])).sum() for i in range(len(alphas))], label=r"$\alpha$-\textsc{FairCB}")
plt.plot(alphas, [(faircb_cumulative_rewards[i][-1] ** (1 - alphas[i]) / (1 - alphas[i])).sum() for i in range(len(alphas))], label=r"\textsc{FairCB}")
plt.legend(loc="upper left", fontsize="large")
plt.xlabel(r"$\alpha$", fontsize="large")
plt.ylabel(r"$\alpha$-fair objective", fontsize="large")
plt.savefig(ALPHA_FAIR_OBJECTIVE_VARIATION_PATH)

plt.figure(2)
plt.plot(alphas, [jains_fairness_index(cumulative_rewards[i][-1]) for i in range(len(alphas))], label=r"$\alpha$-\textsc{FairCB}")
plt.plot(alphas, [jains_fairness_index(faircb_cumulative_rewards[i][-1]) for i in range(len(alphas))], label=r"\textsc{FairCB}")
plt.legend(loc="upper left", fontsize="large")
plt.xlabel(r"$\alpha$", fontsize="large")
plt.ylabel(r"Jain's Fairness Index", fontsize="large")
plt.savefig(FAIRNESS_VARIATION_PATH)
