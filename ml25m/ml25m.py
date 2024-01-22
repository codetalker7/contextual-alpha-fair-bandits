from driver import *

## getting the offline optimal objectives
with open(OFFLINE_OPTIMAL_FILE, "rb") as f:
    offline_optimal_values = pickle.load(f)

## running the policies
from Hedge import Hedge
from ParallelOPF import ParallelOPF
from FairCB import FairCB
from utils import jains_fairness_index

## getting the context distribution for fairCB
valueCounts = data["userId"].value_counts()
context_distribution = np.zeros((NUM_CONTEXTS, ))
for context_id in range(NUM_CONTEXTS):
    context_distribution[context_id] = valueCounts.loc[map_index_to_user[context_id]] / len(data)

policies = [
    Hedge(NUM_CONTEXTS, NUM_ARMS, len(data)),
    ParallelOPF(NUM_CONTEXTS, NUM_ARMS, ALPHA),
    FairCB(NUM_CONTEXTS, NUM_ARMS, config_dict["FAIRCBFAIRNESS"] / NUM_ARMS, len(data), context_distribution)
]

## keeping track of cumulative rewards
cumulative_rewards = [[np.ones(NUM_ARMS, )] for i in range(len(policies))]

## alpha-performance
alpha_performance = [[] for i in range(len(policies))]

## jain's fairness index
fairness_index = [[] for i in range(len(policies))]

## sum of rewards
sum_rewards = [[0] for i in range(len(policies))]

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
        if i == 0:
            # hedge
            cumulative_rewards[i].append(last_cum_rewards + rewards * (policies[i].weights / np.sum(policies[i].weights)))
        else:
            cumulative_rewards[i].append(last_cum_rewards + rewards * policies[i].last_decision)

    ## updating alpha-performance
    for i in range(len(policies)):
        last_cum_rewards = cumulative_rewards[i][-1]
        alpha_performance[i].append((last_cum_rewards ** (1 - ALPHA) / (1 - ALPHA)).sum())

    ## update the fairness index
    for i in range(len(policies)):
        last_cum_rewards = cumulative_rewards[i][-1]
        fairness_index[i].append(jains_fairness_index(last_cum_rewards))

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
import matplotlib.pyplot as plt
USETEXLIVE = config_dict["USETEXLIVE"]

if USETEXLIVE:
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    labels = [r"\textsc{Hedge}", r"$\alpha$\textsc{-FairCB}", r"\textsc{FairCB}"]
else:
    labels = ["Hedge", "alpha-FairCB", "FairCB"]
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams["figure.figsize"] = (5, 4)

PERFORMANCE_PLOT_PATH = "plots/performance_full_information.pdf"
ALPHA_PERFORMANCE_PLOT_PATH = "plots/alpha_performance_full_information.pdf"
JAINS_FAIRNESS_PLOT_PATH = "plots/jains_index_full_information.pdf"
APPROXIMATE_REGRET_PLOT_PATH = "plots/approximate_regret_full_information.pdf"
STANDARD_REGRET_PLOT_PATH = "plots/standard_regret_full_information.pdf"

time = np.arange(1, len(data) + 1)

## plotting performance
performance = [np.array(sum_rewards[i][1:]) * (1 / time) for i in range(len(policies))]

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
