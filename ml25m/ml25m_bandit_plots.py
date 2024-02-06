from driver_bandit_information import *

with open(BANDIT_INFORMATION_FILE, 'rb') as f:
    bandit_information_dict = pickle.load(f)
    policies = bandit_information_dict["policies"]
    fairness_index = bandit_information_dict["fairness_index"]
    cumulative_rewards = bandit_information_dict["cumulative_rewards"]

    if PLOTREGRET:
        standard_regrets = bandit_information_dict["standard_regrets"]
        approximate_regrets = bandit_information_dict["approximate_regrets"]

# %matplotlib inline

if USETEXLIVE:
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    labels = [r"\text{Putta and Aggarwal 2022}", r"$\alpha$\textsc{-FairCB}"]
else:
    labels = ["Putta and Aggarwal 2022", "alpha-FairCB"]
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