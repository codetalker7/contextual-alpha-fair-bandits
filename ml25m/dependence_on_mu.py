from driver import *

NUM_POLICIES = 51

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

FAIRNESS_VALUES_FILE = f'pickled_files/fairness_values_rows={config_dict["ROWS"]}_seed={SEED}_alpha={ALPHA}_smallreward={SMALL_REWARD}_usetimestamps={USETIMESTAMPS}_frequency={config_dict["FREQUENCY"]}_highfrequency={config_dict["HIGHFREQUENCY"]}_numpolicies={NUM_POLICIES}.pickle'
fairness_values = [fairness_index[i][len(data) - 1] for i in range(len(policies) - 1)]
alphaFairCBValue = fairness_index[len(policies) - 1][len(data) - 1]
with open(FAIRNESS_VALUES_FILE, 'wb') as f:
    pickle.dump({"fairness_values": fairness_values, "alphaFairCBValue": alphaFairCBValue}, f)

# ## plotting the fairness levels
# import matplotlib.pyplot as plt
# USETEXLIVE = config_dict["USETEXLIVE"]

# if USETEXLIVE:
#     plt.rc('text', usetex=True)
#     plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
# plt.style.use('seaborn-v0_8-darkgrid')
# plt.rcParams["figure.figsize"] = (5, 4)

# JAINS_FAIRNESS_PLOT_PATH = "plots/jains_index_varying_nus.pdf"

# plt.figure(0)
# ax = plt.axes()
# plt.plot(nus, fairness_values)
# plt.axhline(y=fairness_index[NUM_POLICIES - 1][len(data) - 1])
# plt.xlabel("nu", fontsize="large")
# plt.ylabel("Jain's Fairness Index", fontsize="large")
# # if USETEXLIVE:
# #     plt.text(0.5, 0.95, f"$\\alpha={ALPHA}$, $\\nu = {config_dict['FAIRCBFAIRNESS']}$, $\\delta={SMALL_REWARD}$", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
# # plt.title("Jain's Fairness Index Plot (Full Information Setting)", fontsize="large")
# plt.savefig(JAINS_FAIRNESS_PLOT_PATH, bbox_inches='tight', pad_inches=0.01)


