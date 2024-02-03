from driver import *

NUM_POLICIES = 101
PLOTREGRET = config_dict["PLOTREGRET"]

## getting the offline optimal objectives
if PLOTREGRET:
    with open(OFFLINE_OPTIMAL_FILE, "rb") as f:
        offline_optimal_values = pickle.load(f)

from ParallelOPF import ParallelOPF
from FairCB import FairCB
from utils import jains_fairness_index

## getting the context distribution for fairCB
valueCounts = data["userId"].value_counts()
context_distribution = np.zeros((NUM_CONTEXTS, ))
for context_id in range(NUM_CONTEXTS):
    context_distribution[context_id] = valueCounts.loc[map_index_to_user[context_id]] / len(data)

## will have one FairCB policy for each nu
nus = np.linspace(0, 1, NUM_POLICIES).tolist()[:-1]
policies = [
    FairCB(NUM_CONTEXTS, NUM_ARMS, nus[i] / NUM_ARMS, len(data), context_distribution)
    for i in range(len(nus))
]
policies.append(ParallelOPF(NUM_CONTEXTS, NUM_ARMS, ALPHA)) # last policy is our policy

## keeping track of cumulative rewards
cumulative_rewards = [[np.ones(NUM_ARMS, )] for i in range(len(policies))]

## alpha-performance
alpha_performance = [[] for i in range(len(policies))]

## jain's fairness index
fairness_index = [[] for i in range(len(policies))]

## sum of rewards
sum_rewards = [[0] for i in range(len(policies))]

if PLOTREGRET:
    ## standard regrets
    standard_regrets = [[] for i in range(len(policies))]

    ## approximate regrets
    approximate_regrets = [[] for i in range(len(policies))]

for t in tqdm(range(len(data))):
    data_point = data.iloc[t]
    userId = int(data_point["userId"])
    movieId = int(data_point["movieId"])

    recommended_genres = [policies[i].decision(map_user_to_index[userId]) for i in range(len(policies))]

    ## get rewards corresponding to the movie
    rewards = reward_function(movieId)

    ## update performance
    for i in range(len(policies)):
        sum_rewards[i].append(sum_rewards[i][-1] + rewards[recommended_genres[i] - 1])

    ## update cumulative rewards
    for i in range(len(policies)):
        last_cum_rewards = cumulative_rewards[i][-1]
        cumulative_rewards[i].append(last_cum_rewards + rewards * policies[i].last_decision)

    ## updating alpha-performance
    for i in range(len(policies)):
        last_cum_rewards = cumulative_rewards[i][-1]
        alpha_performance[i].append((last_cum_rewards ** (1 - ALPHA) / (1 - ALPHA)).sum())

    ## update the fairness index
    for i in range(len(policies)):
        last_cum_rewards = cumulative_rewards[i][-1]
        fairness_index[i].append(jains_fairness_index(last_cum_rewards))

    if PLOTREGRET:
        ## update the standard regrets
        for i in range(len(policies)):
            last_cum_rewards = cumulative_rewards[i][-1]
            standard_regrets[i].append(offline_optimal_values[t] - (last_cum_rewards ** (1 - ALPHA) / (1 - ALPHA)).sum())
        ## update the approximate regret
        for i in range(len(policies)):
            last_cum_rewards = cumulative_rewards[i][-1]
            approximate_regrets[i].append(offline_optimal_values[t] - APPROX_FACTOR * (last_cum_rewards ** (1 - ALPHA) / (1 - ALPHA)).sum())

    ## feedback rewards to the policies
    for i in range(len(policies)):
        policies[i].feedback(rewards)