from driver_full_information import *

alphas = np.linspace(0, 1, NUM_ALPHAS + 1).tolist()[:-1]

policies = [ParallelOPF(NUM_CONTEXTS, NUM_ARMS, alphas[i]) for i in range(len(alphas))]

cumulative_rewards = [[np.ones(NUM_ARMS, )] for i in range(len(alphas))]

for t in tqdm(range(len(data))):
    data_point = data.iloc[t]
    userId = int(data_point["userId"])
    movieId = int(data_point["movieId"])

    # map user id to an index in the range [0, NUM_CONTEXTS - 1]
    userId = map_user_to_index[userId]

    recommended_genres = [policies[i].decision(userId) for i in range(len(alphas))]

    ## get rewards corresponding to the movie
    rewards = get_rewards(movieId)

    ## update cumulative rewards
    for i in range(len(alphas)):
        last_cum_rewards = cumulative_rewards[i][-1]
        cumulative_rewards[i].append(last_cum_rewards + rewards * policies[i].last_decision)

    ## feedback rewards to the policies
    for i in range(len(alphas)):
        policies[i].feedback(rewards)

## saving the cumulative rewards as pickle files
with open("cumulative_rewards.pickle", 'wb') as f:
    pickle.dump({
        "cumulative_rewards": cumulative_rewards,
        "alphas": alphas,
    },
    f
)