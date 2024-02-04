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

## plotting
# %matplotlib inline

if USETEXLIVE:
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    labels = [r"\textsc{Hedge}", r"$\alpha$\textsc{-FairCB}", r"\textsc{FairCB}"]
else:
    labels = ["Hedge", "alpha-FairCB", "FairCB"]
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams["figure.figsize"] = (5, 4)

time = np.arange(1, len(data) + 1)

## plotting fairness
plt.figure(0)
ax = plt.axes()
for i in range(len(policies)):
    plt.plot(time, fairness_index[i], label=labels[i])
plt.legend(loc="lower right", fontsize="large")
plt.xlabel("Time", fontsize="large")
plt.ylabel("Jain's Fairness Index", fontsize="large")
# if USETEXLIVE:
#     plt.text(0.5, 0.95, f"$\\alpha={ALPHA}$, $\\nu = {config_dict['FAIRCBFAIRNESS']}$, $\\delta={SMALL_REWARD}$", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
# plt.title("Jain's Fairness Index Plot (Full Information Setting)", fontsize="large")
plt.savefig(JAINS_FAIRNESS_PLOT_PATH, bbox_inches='tight', pad_inches=0.01)

## plotting standard regrets
if PLOTREGRET:
    plt.figure(1)
    ax = plt.axes()
    for i in range(len(policies)):
        plt.plot(time, standard_regrets[i], label=labels[i])
    plt.legend(loc="upper left", fontsize="large")
    plt.xlabel("Time", fontsize="large")
    plt.ylabel("Standard Regret", fontsize="large")
    # if USETEXLIVE:
    #     plt.text(0.5, 0.95, f"$\\alpha={ALPHA}$, $\\nu = {config_dict['FAIRCBFAIRNESS']}$, $\\delta={SMALL_REWARD}$", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    # plt.title("Standard Regret Plot (Full Information Setting)", fontsize="large")
    plt.savefig(STANDARD_REGRET_PLOT_PATH, bbox_inches='tight', pad_inches=0.01)

    ## plotting approximate regrets
    plt.figure(2)
    ax = plt.axes()
    for i in range(len(policies)):
        plt.plot(time, approximate_regrets[i], label=labels[i])
    plt.legend(loc="upper left", fontsize="large")
    plt.xlabel("Time", fontsize="large")
    plt.ylabel("Approximate Regret", fontsize="large")
    # if USETEXLIVE:
    #     plt.text(0.5, 0.95, f"$\\alpha={ALPHA}$, $\\nu = {config_dict['FAIRCBFAIRNESS']}$, $\\delta={SMALL_REWARD}$", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    # plt.title("Approximate Regret Plot (Full Information Setting)", fontsize="large")
    plt.savefig(APPROXIMATE_REGRET_PLOT_PATH, bbox_inches='tight', pad_inches=0.01)
