from driver import *

PLOTREGRET = config_dict["PLOTREGRET"]

## getting the offline optimal objectives
if PLOTREGRET:
    with open(OFFLINE_OPTIMAL_FILE, "rb") as f:
        offline_optimal_values = pickle.load(f)

## running the policies
from ScaleFreeMAB import ScaleFreeMAB
from ParallelScaleFreeMAB import ParallelScaleFreeMAB
from utils import jains_fairness_index

policies = [
    ScaleFreeMAB(NUM_CONTEXTS, NUM_ARMS),
    ParallelScaleFreeMAB(NUM_CONTEXTS, NUM_ARMS, ALPHA)
]

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

    ## characteristic vector for chosen arm
    char_vectors = [np.zeros(NUM_ARMS, ) for i in range(len(policies))]
    for i in range(len(policies)):
        char_vectors[i][recommended_genres[i] - 1] = 1

    ## get rewards corresponding to the movie
    rewards = get_rewards(movieId)

    ## update performance
    for i in range(len(policies)):
        sum_rewards[i].append(sum_rewards[i][-1] + rewards[recommended_genres[i] - 1])
    
    ## update cum rewards
    for i in range(len(policies)):
        last_cum_rewards = cumulative_rewards[i][-1]
        cumulative_rewards[i].append(last_cum_rewards + rewards * char_vectors[i])

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
        policies[i].feedback(rewards[recommended_genres[i] - 1])

## plotting
# %matplotlib inline
import matplotlib.pyplot as plt
USETEXLIVE = config_dict["USETEXLIVE"]

if USETEXLIVE:
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    labels = [r"\text{Putta and Aggarwal 2022}", r"$\alpha$\textsc{-FairCB}"]
else:
    labels = ["Putta and Aggarwal 2022", "alpha-FairCB"]
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams["figure.figsize"] = (5, 4)

PERFORMANCE_PLOT_PATH = "plots/performance_bandit_information.pdf"
ALPHA_PERFORMANCE_PLOT_PATH = "plots/alpha_performance_bandit_information.pdf"
JAINS_FAIRNESS_PLOT_PATH = "plots/jains_index_bandit_information.pdf"
STANDARD_REGRET_PLOT_PATH = "plots/standard_regret_bandit_information.pdf"
APPROXIMATE_REGRET_PLOT_PATH = "plots/approximate_regret_bandit_information.pdf"

time = np.arange(1, len(data) + 1)

## plotting performance
performance = [np.array(sum_rewards[i][1:]) * (1 / time) for i in range(len(policies))]
# scaleFree_performance = np.array(scaleFree_sum_rewards)[1:] * (1 / time)
# parallelScaleFree_performance = np.array(parallelScaleFree_sum_rewards)[1:] * (1 / time)

plt.figure(0)
ax = plt.axes()
for i in range(len(policies)):
    plt.plot(time, performance[i], label=labels[i])
plt.xlabel("Time")
plt.ylabel("Performance")
# if USETEXLIVE:
#     plt.text(0.5, 0.95, f"$\\alpha={ALPHA}$, $\\nu = {config_dict['FAIRCBFAIRNESS']}$, $\\delta={SMALL_REWARD}$", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
# plt.title("Performance Plot")
plt.savefig(PERFORMANCE_PLOT_PATH, bbox_inches='tight', pad_inches=0.01)

## plotting alpha-performance
plt.figure(1)
ax = plt.axes()
for i in range(len(policies)):
    plt.plot(time, alpha_performance[i], label=labels[i])
plt.legend(loc="upper left", fontsize="large")
plt.xlabel("Time", fontsize="large")
if USETEXLIVE:
    # plt.text(0.5, 0.95, f"$\\alpha={ALPHA}$, $\\nu = {config_dict['FAIRCBFAIRNESS']}$, $\\delta={SMALL_REWARD}$", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    plt.ylabel(r"$\alpha$-Performance", fontsize="large")
else:
    plt.ylabel("alpha-Performance", fontsize="large")
# plt.title("Alpha-Performance Plot (Full Information Setting)", fontsize="large")
plt.savefig(ALPHA_PERFORMANCE_PLOT_PATH, bbox_inches='tight', pad_inches=0.01)

## plotting fairness
plt.figure(2)
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
    plt.figure(3)
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
    plt.figure(4)
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
