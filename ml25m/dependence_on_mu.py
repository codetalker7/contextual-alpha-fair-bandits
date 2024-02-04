from driver_full_information import *

## will have one FairCB policy for each nu
nus = np.linspace(0, 1, NUM_NUS + 1).tolist()[:-1]
policies = [
    FairCB(NUM_CONTEXTS, NUM_ARMS, nus[i] / NUM_ARMS, len(data), context_distribution)
    for i in range(len(nus))
]
policies.append(ParallelOPF(NUM_CONTEXTS, NUM_ARMS, ALPHA)) # last policy is our policy

## keeping track of cumulative rewards
cumulative_rewards = [[np.ones(NUM_ARMS, )] for i in range(len(policies))]

## jain's fairness index
fairness_index = [[] for i in range(len(policies))]

for t in tqdm(range(len(data))):
    data_point = data.iloc[t]
    userId = int(data_point["userId"])
    movieId = int(data_point["movieId"])

    recommended_genres = [policies[i].decision(map_user_to_index[userId]) for i in range(len(policies))]

    ## get rewards corresponding to the movie
    rewards = reward_function(movieId)

    ## update cumulative rewards
    for i in range(len(policies)):
        last_cum_rewards = cumulative_rewards[i][-1]
        cumulative_rewards[i].append(last_cum_rewards + rewards * policies[i].last_decision)

    ## update the fairness index
    for i in range(len(policies)):
        last_cum_rewards = cumulative_rewards[i][-1]
        fairness_index[i].append(jains_fairness_index(last_cum_rewards))

    ## feedback rewards to the policies
    for i in range(len(policies)):
        policies[i].feedback(rewards)

fairness_values = [fairness_index[i][len(data) - 1] for i in range(len(policies) - 1)]
alphaFairCBValue = fairness_index[len(policies) - 1][len(data) - 1]
with open(FAIRNESS_VALUES_FILE, 'wb') as f:
    pickle.dump({
        "fairness_values": fairness_values,
        "alphaFairCBValue": alphaFairCBValue,
        "nus": nus
        },
        f
    )
