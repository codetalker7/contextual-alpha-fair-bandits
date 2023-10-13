from driver import *

from ScaleFreeMAB import ScaleFreeMAB
from ParallelScaleFreeMAB import ParallelScaleFreeMAB
from utils import jains_fairness_index

scaleFreePolicy = ScaleFreeMAB(NUM_CONTEXTS, NUM_ARMS)
parallelScaleFreePolicy = ParallelScaleFreeMAB(NUM_CONTEXTS, NUM_ARMS, ALPHA)

## keeping track of cumulative rewards
scaleFree_cum_rewards = [np.ones((NUM_ARMS, ))]
parallelScaleFree_cum_rewards = [np.ones((NUM_ARMS, ))]

## alpha-performance
scaleFree_alpha_performance = []
parallelScaleFree_alpha_performance = []

## jain's fairness index
scaleFree_fairness_index = []
parallelScaleFree_fairness_index = []

## sum of rewards
scaleFree_sum_rewards = [0]
parallelScaleFree_sum_rewards = [0]

## standard regrets
scaleFree_standard_regret = []
parallelScaleFree_standard_regret = []

## approximate regrets
scaleFree_approximate_regret = []
parallelScaleFree_approximate_regret = []

for t in tqdm(range(len(data))):
    data_point = data.iloc[t]
    userId = int(data_point["userId"])
    movieId = int(data_point["movieId"])

    scaleFree_recommended_genre = scaleFreePolicy.decision(userId - 1)      # context labels start from 0
    parallelScaleFree_recommended_genre = parallelScaleFreePolicy.decision(userId - 1)

    ## characteristic vector for chosen arm
    scaleFree_char_vector = np.zeros((NUM_ARMS, ))
    scaleFree_char_vector[scaleFree_recommended_genre - 1] = 1

    parallelScaleFree_char_vector = np.zeros((NUM_ARMS, ))
    parallelScaleFree_char_vector[parallelScaleFree_recommended_genre - 1] = 1

    ## get rewards corresponding to the movie
    rewards = get_rewards(movieId)

    ## update performance
    scaleFree_sum_rewards.append(scaleFree_sum_rewards[-1] + rewards[scaleFree_recommended_genre - 1])
    parallelScaleFree_sum_rewards.append(parallelScaleFree_sum_rewards[-1] + rewards[parallelScaleFree_recommended_genre - 1])

    ## update cum rewards
    scaleFree_last_cum_rewards = scaleFree_cum_rewards[-1]
    parallelScaleFree_last_cum_rewards = parallelScaleFree_cum_rewards[-1]

    scaleFree_cum_rewards.append(scaleFree_last_cum_rewards + rewards * scaleFree_char_vector)
    parallelScaleFree_cum_rewards.append(parallelScaleFree_last_cum_rewards + rewards * parallelScaleFree_char_vector)

    ## updating alpha-performance
    scaleFree_alpha_performance.append((scaleFree_cum_rewards[-1] ** (1 - ALPHA) / (1 - ALPHA)).sum()) 
    parallelScaleFree_alpha_performance.append((parallelScaleFree_cum_rewards[-1] ** (1 - ALPHA) / (1 - ALPHA)).sum())

    ## update the fairness index
    scaleFree_fairness_index.append(jains_fairness_index(scaleFree_cum_rewards[-1]))
    parallelScaleFree_fairness_index.append(jains_fairness_index(parallelScaleFree_cum_rewards[-1]))

    ## update the standard regrets
    scaleFree_standard_regret.append(offline_optimal_values[t] - ((scaleFree_cum_rewards[-1] ** (1 - ALPHA)) / (1 - ALPHA)).sum())
    parallelScaleFree_standard_regret.append(offline_optimal_values[t] - ((parallelScaleFree_cum_rewards[-1] ** (1 - ALPHA)) / (1 - ALPHA)).sum())

    ## update the approximate regrets
    scaleFree_approximate_regret.append(offline_optimal_values[t] - APPROX_FACTOR * ((scaleFree_cum_rewards[-1] ** (1 - ALPHA)) / (1 - ALPHA)).sum())
    parallelScaleFree_approximate_regret.append(offline_optimal_values[t] - APPROX_FACTOR * ((parallelScaleFree_cum_rewards[-1] ** (1 - ALPHA)) / (1 - ALPHA)).sum())

    ## feedback rewards to the policies
    scaleFreePolicy.feedback(rewards[scaleFree_recommended_genre - 1])
    parallelScaleFreePolicy.feedback(rewards[parallelScaleFree_recommended_genre - 1])

## plotting
# %matplotlib inline
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams["figure.figsize"] = (5, 4)

PERFORMANCE_PLOT_PATH = "plots/performance_bandit_information.pdf"
ALPHA_PERFORMANCE_PLOT_PATH = "plots/alpha_performance_bandit_information.pdf"
JAINS_FAIRNESS_PLOT_PATH = "plots/jains_index_bandit_information.pdf"
STANDARD_REGRET_PLOT_PATH = "plots/standard_regret_bandit_information.pdf"
APPROXIMATE_REGRET_PLOT_PATH = "plots/approximate_regret_bandit_information.pdf"

time = np.arange(1, len(data) + 1)

## plotting performance
scaleFree_performance = np.array(scaleFree_sum_rewards)[1:] * (1 / time)
parallelScaleFree_performance = np.array(parallelScaleFree_sum_rewards)[1:] * (1 / time)

plt.figure(0)
plt.plot(time, scaleFree_performance, label="Putta \& Aggarwal, 2022")
plt.plot(time, parallelScaleFree_performance, label=r"$\alpha\textsc{-FairCB}$")
plt.legend(loc="upper left", fontsize="large")
plt.xlabel("Time", fontsize="large")
plt.ylabel("Performance", fontsize="large")
plt.savefig(PERFORMANCE_PLOT_PATH)

## plotting alpha-performance
plt.figure(1)
plt.plot(time, scaleFree_alpha_performance, label="Putta \& Aggarwal, 2022")
plt.plot(time, parallelScaleFree_alpha_performance, label=r"$\alpha\textsc{-FairCB}$")
plt.legend(loc="upper left", fontsize="large")
plt.xlabel("Time", fontsize="large")
plt.ylabel(r'$\alpha$-Performance', fontsize="large")
plt.savefig(ALPHA_PERFORMANCE_PLOT_PATH, bbox_inches='tight', pad_inches=0.01)

## plotting fairness
plt.figure(2)
plt.plot(time, scaleFree_fairness_index, label="Putta \& Aggarwal, 2022")
plt.plot(time, parallelScaleFree_fairness_index, label=r"$\alpha\textsc{-FairCB}$")
plt.legend(loc="lower left", fontsize="large")
plt.xlabel("Time", fontsize="large")
plt.ylabel("Jain's Fairness Index", fontsize="large")
plt.savefig(JAINS_FAIRNESS_PLOT_PATH, bbox_inches='tight', pad_inches=0.01)

## plotting standard regrets
plt.figure(3)
plt.plot(time, scaleFree_standard_regret, label="Putta \& Aggarwal, 2022")
plt.plot(time, parallelScaleFree_standard_regret, label=r"$\alpha\textsc{-FairCB}$")
plt.legend(loc="center right", fontsize="large")
plt.xlabel("Time", fontsize="large")
plt.ylabel("Standard Regret", fontsize="large")
plt.savefig(STANDARD_REGRET_PLOT_PATH, bbox_inches='tight', pad_inches=0.01)

## plotting approximate regrets
plt.figure(4)
plt.plot(time, scaleFree_approximate_regret, label="Putta \& Aggarwal, 2022")
plt.plot(time, parallelScaleFree_approximate_regret, label=r"$\alpha\textsc{-FairCB}$")
plt.legend(loc="center right", fontsize="large")
plt.xlabel("Time", fontsize="large")
plt.ylabel("Approximate Regret", fontsize="large")
plt.savefig(APPROXIMATE_REGRET_PLOT_PATH, bbox_inches='tight', pad_inches=0.01)
