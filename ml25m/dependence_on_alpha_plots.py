from driver_full_information import *

## loading the cumulative rewards
with open(CUMULATIVE_REWARDS_FILE, 'rb') as f:
    cumulative_rewards_dict = pickle.load(f)
    cumulative_rewards = cumulative_rewards_dict["cumulative_rewards"]
    alphas = cumulative_rewards_dict["alphas"]

if USETEXLIVE:
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams["figure.figsize"] = (5, 4)

plt.figure(0)
ax = plt.axes()
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth(1)  
ax.tick_params(direction='out', length=5, width=0.5, grid_alpha=0.5)
plt.plot(alphas, [cumulative_rewards[i][-1].sum() / NUM_ARMS for i in range(len(alphas))])
plt.legend(loc="upper left", fontsize="large")
if USETEXLIVE: 
    plt.xlabel(r"$\alpha$", fontsize="large")
else:
    plt.xlabel("alpha", fontsize="large")
plt.ylabel("Average Cumulative Rewards", fontsize="large")
plt.savefig(AVERAGED_CUMULATIVE_REWARDS_PATH)

plt.figure(2)
ax = plt.axes()
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth(1)  
ax.tick_params(direction='out', length=5, width=0.5, grid_alpha=0.5)
plt.plot(alphas, [jains_fairness_index(cumulative_rewards[i][-1]) for i in range(len(alphas))], color='b')
plt.legend(loc="upper left", fontsize="large")
if USETEXLIVE: 
    plt.xlabel(r"$\alpha$", fontsize="large")
else:
    plt.xlabel("alpha", fontsize="large")
plt.ylabel(r"Jain's Fairness Index", fontsize="large")
plt.savefig(VARYING_ALPHAS_PLOT_PATH)