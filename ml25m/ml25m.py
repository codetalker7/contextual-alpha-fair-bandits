from driver_full_information import *

## getting the offline optimal objectives
if PLOTREGRET:
    with open(OFFLINE_OPTIMAL_FILE, "rb") as f:
        offline_optimal_values = pickle.load(f)

policies = [
    Hedge(NUM_CONTEXTS, NUM_ARMS, len(data)),
    ParallelOPF(NUM_CONTEXTS, NUM_ARMS, ALPHA),
    FairCB(NUM_CONTEXTS, NUM_ARMS, config_dict["FAIRCBFAIRNESS"] / NUM_ARMS, len(data), context_distribution)
]

## keeping track of cumulative rewards
cumulative_rewards = [[np.ones(NUM_ARMS, )] for i in range(len(policies))]

## jain's fairness index
fairness_index = [[] for i in range(len(policies))]

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

    ## update cumulative rewards
    for i in range(len(policies)):
        last_cum_rewards = cumulative_rewards[i][-1]
        if i == 0:
            # hedge
            cumulative_rewards[i].append(last_cum_rewards + rewards * (policies[i].weights / np.sum(policies[i].weights)))
        else:
            cumulative_rewards[i].append(last_cum_rewards + rewards * policies[i].last_decision)

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

with open(FULL_INFORMATION_FILE, 'wb') as f:
    full_information_dict = {
        "policies": policies, 
        "fairness_index": fairness_index,
        "cumulative_rewards": cumulative_rewards
    }
    if PLOTREGRET:
        full_information_dict["standard_regrets"] = standard_regrets
        full_information_dict["approximate_regrets"] = approximate_regrets
    pickle.dump(full_information_dict, f)