from driver import *

## running the policies
from Hedge import Hedge
from ParallelOPF import ParallelOPF
from FairCB import FairCB
from utils import jains_fairness_index


## getting the context distribution for fairCB
valueCounts = data["userId"].value_counts()
context_distribution = np.zeros((NUM_CONTEXTS, ))
for context_id in range(NUM_CONTEXTS):
    context_distribution[context_id] = valueCounts.loc[context_id + 1] / len(data)

## initialize policies
hedge = Hedge(NUM_CONTEXTS, NUM_ARMS, len(data))
parallelOPF = ParallelOPF(NUM_CONTEXTS, NUM_ARMS, ALPHA)
fairCB = FairCB(NUM_CONTEXTS, NUM_ARMS, 1 / (2 * NUM_ARMS), len(data), context_distribution)

## keeping track of cumulative rewards
hedge_cum_rewards = [np.ones((NUM_ARMS, ))]
popf_cum_rewards = [np.ones((NUM_ARMS, ))]
fairCB_cum_rewards = [np.ones((NUM_ARMS, ))]

## alpha-performance
hedge_alpha_performance = []
popf_alpha_performance = []
fairCB_alpha_performance = []

## jain's fairness index
hedge_fairness_index = []
popf_fairness_index = []
fairCB_fairness_index = []

## sum of rewards
hedge_sum_rewards = [0]
popf_sum_rewards = [0]
fairCB_sum_rewards = [0]

## standard regrets
hedge_standard_regret = []
popf_standard_regret = []
fairCB_standard_regret = []

## approximate regrets
hedge_approximate_regret = []
popf_approximate_regret = []
fairCB_approximate_regret = []

for t in tqdm(range(len(data))):
    data_point = data.iloc[t]
    userId = int(data_point["userId"])
    movieId = int(data_point["movieId"])

    hedge_recommended_genre = hedge.decision(userId - 1)    # context labels start from 0
    popf_recommended_genre = parallelOPF.decision(userId - 1)
    fairCB_recommended_genre = fairCB.decision(userId - 1)

    ## get rewards corresponding to the movie
    rewards = get_rewards(movieId)

    ## update performance
    hedge_sum_rewards.append(hedge_sum_rewards[-1] + rewards[hedge_recommended_genre - 1])
    popf_sum_rewards.append(popf_sum_rewards[-1] + rewards[popf_recommended_genre - 1])
    fairCB_sum_rewards.append(fairCB_sum_rewards[-1] + rewards[fairCB_recommended_genre - 1])

    ## update cum rewards
    hedge_last_cum_rewards = hedge_cum_rewards[-1]
    popf_last_cum_rewards = popf_cum_rewards[-1]
    fairCB_last_cum_rewards = fairCB_cum_rewards[-1]

    hedge_cum_rewards.append(hedge_last_cum_rewards + rewards * (hedge.weights / np.sum(hedge.weights)))
    popf_cum_rewards.append(popf_last_cum_rewards + rewards * parallelOPF.last_decision)
    fairCB_cum_rewards.append(fairCB_last_cum_rewards + rewards * fairCB.last_decision)

    ## updating alpha-performance
    hedge_alpha_performance.append((hedge_cum_rewards[-1] ** (1 - ALPHA) / (1 - ALPHA)).sum())
    popf_alpha_performance.append((popf_cum_rewards[-1] ** (1 - ALPHA) / (1 - ALPHA)).sum())
    fairCB_alpha_performance.append((fairCB_cum_rewards[-1] ** (1 - ALPHA) / (1 - ALPHA)).sum())

    ## update the fairness index
    hedge_fairness_index.append(jains_fairness_index(hedge_cum_rewards[-1]))
    popf_fairness_index.append(jains_fairness_index(popf_cum_rewards[-1]))
    fairCB_fairness_index.append(jains_fairness_index(fairCB_cum_rewards[-1]))

    ## update the standard regrets
    hedge_standard_regret.append(offline_optimal_values[t] - ((hedge_cum_rewards[-1] ** (1 - ALPHA)) / (1 - ALPHA)).sum())
    popf_standard_regret.append(offline_optimal_values[t] - ((popf_cum_rewards[-1] ** (1 - ALPHA)) / (1 - ALPHA)).sum())
    fairCB_standard_regret.append(offline_optimal_values[t] - ((fairCB_cum_rewards[-1] ** (1 - ALPHA)) / (1 - ALPHA)).sum())

    ## update the approximate regret
    hedge_approximate_regret.append(offline_optimal_values[t] - APPROX_FACTOR * ((hedge_cum_rewards[-1] ** (1 - ALPHA)) / (1 - ALPHA)).sum())
    popf_approximate_regret.append(offline_optimal_values[t] - APPROX_FACTOR * ((popf_cum_rewards[-1] ** (1 - ALPHA)) / (1 - ALPHA)).sum())
    fairCB_approximate_regret.append(offline_optimal_values[t] - APPROX_FACTOR * ((fairCB_cum_rewards[-1] ** (1 - ALPHA)) / (1 - ALPHA)).sum())

    ## feedback rewards to the policies
    hedge.feedback(rewards)
    parallelOPF.feedback(rewards)
    fairCB.feedback(rewards)

## plotting
# %matplotlib inline
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams["figure.figsize"] = (5, 4)

PERFORMANCE_PLOT_PATH = "plots/performance_full_information.pdf"
ALPHA_PERFORMANCE_PLOT_PATH = "plots/alpha_performance_full_information.pdf"
JAINS_FAIRNESS_PLOT_PATH = "plots/jains_index_full_information.pdf"
APPROXIMATE_REGRET_PLOT_PATH = "plots/approximate_regret_full_information.pdf"
STANDARD_REGRET_PLOT_PATH = "plots/standard_regret_full_information.pdf"

time = np.arange(1, len(data) + 1)

## plotting performance
hedge_performance = np.array(hedge_sum_rewards)[1:] * (1 / time)
popf_performance = np.array(popf_sum_rewards)[1:] * (1 / time)
fairCB_performance = np.array(fairCB_sum_rewards)[1:] * (1 / time)

plt.figure(0)
plt.plot(time, hedge_performance, label=r"\textsc{Hedge}")
plt.plot(time, popf_performance, label=r"$\alpha$\textsc{-FairCB}")
plt.plot(time, fairCB_performance, label=r"\textsc{FairCB}")
plt.xlabel("Time")
plt.ylabel("Performance")
# plt.title("Performance Plot")
plt.savefig(PERFORMANCE_PLOT_PATH, bbox_inches='tight', pad_inches=0.01)

## plotting alpha-performance
plt.figure(1)
plt.plot(time, hedge_alpha_performance, label=r"\textsc{Hedge}")
plt.plot(time, popf_alpha_performance, label=r"$\alpha$\textsc{-FairCB}")
plt.plot(time, fairCB_alpha_performance, label=r"\textsc{FairCB}")
plt.legend(loc="upper left", fontsize="large")
plt.xlabel("Time", fontsize="large")
plt.ylabel(r"$\alpha$-Performance", fontsize="large")
# plt.title("Alpha-Performance Plot (Full Information Setting)", fontsize="large")
plt.savefig(ALPHA_PERFORMANCE_PLOT_PATH, bbox_inches='tight', pad_inches=0.01)

## plotting fairness
plt.figure(2)
plt.plot(time, hedge_fairness_index, label=r"\textsc{Hedge}")
plt.plot(time, popf_fairness_index, label=r"$\alpha$\textsc{-FairCB}")
plt.plot(time, fairCB_fairness_index, label=r"\textsc{FairCB}")
plt.legend(loc="lower right", fontsize="large")
plt.xlabel("Time", fontsize="large")
plt.ylabel("Jain's Fairness Index", fontsize="large")
# plt.title("Jain's Fairness Index Plot (Full Information Setting)", fontsize="large")
plt.savefig(JAINS_FAIRNESS_PLOT_PATH, bbox_inches='tight', pad_inches=0.01)

## plotting standard regrets
plt.figure(3)
plt.plot(time, hedge_standard_regret, label=r"\textsc{Hedge}")
plt.plot(time, popf_standard_regret, label=r"$\alpha$\textsc{-FairCB}")
plt.plot(time, fairCB_standard_regret, label=r"\textsc{FairCB}")
plt.legend(loc="upper left", fontsize="large")
plt.xlabel("Time", fontsize="large")
plt.ylabel("Standard Regret", fontsize="large")
# plt.title("Standard Regret Plot (Full Information Setting)", fontsize="large")
plt.savefig(STANDARD_REGRET_PLOT_PATH, bbox_inches='tight', pad_inches=0.01)

## plotting approximate regrets
plt.figure(4)
plt.plot(time, hedge_approximate_regret, label=r"\textsc{Hedge}")
plt.plot(time, popf_approximate_regret, label=r"$\alpha$\textsc{-FairCB}")
plt.plot(time, fairCB_approximate_regret, label=r"\textsc{FairCB}")
plt.legend(loc="upper left", fontsize="large")
plt.xlabel("Time", fontsize="large")
plt.ylabel("Approximate Regret", fontsize="large")
# plt.title("Approximate Regret Plot (Full Information Setting)", fontsize="large")
plt.savefig(APPROXIMATE_REGRET_PLOT_PATH, bbox_inches='tight', pad_inches=0.01)
