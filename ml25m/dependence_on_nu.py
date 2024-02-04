from driver_full_information import *

nus = np.linspace(0, 1, NUM_NUS + 1).tolist()[:-1]

## final values
fairness_values = []
alphaFairCBValue = None

for rounds in tqdm(range(VARYING_NU_ROUNDS)):
    ## will have one FairCB policy for each nu
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

    # update the values
    if len(fairness_values) == 0:
        fairness_values = [fairness_index[i][len(data) - 1] for i in range(len(policies) - 1)]
        alphaFairCBValue = fairness_index[len(policies) - 1][len(data) - 1]
    
    else:
        # take the max
        fairness_values = [max(fairness_index[i][len(data) - 1], fairness_values[i]) for i in range(len(policies) - 1)]
        alphaFairCBValue = max(alphaFairCBValue, fairness_index[len(policies) - 1][len(data) - 1])

with open(FAIRNESS_VALUES_FILE, 'wb') as f:
    pickle.dump({
        "fairness_values": fairness_values,
        "alphaFairCBValue": alphaFairCBValue,
        "nus": nus
        },
        f
    )
