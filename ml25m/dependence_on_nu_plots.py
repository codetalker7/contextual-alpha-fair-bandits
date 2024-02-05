from driver_full_information import *

## loading the fairness values
with open(FAIRNESS_VALUES_FILE, 'rb') as f:
    fairness_dict = pickle.load(f)
    nus = fairness_dict["nus"]
    fairness_values = fairness_dict["fairness_values"]
    alphaFairCBValue = fairness_dict["alphaFairCBValue"]

if USETEXLIVE:
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams["figure.figsize"] = (5, 4)

plt.figure(0)
ax = plt.axes()
plt.plot(nus, fairness_values, color='b')
plt.axhline(y=alphaFairCBValue, color='r')
if USETEXLIVE:
    plt.xlabel(r"$\nu$", fontsize="large")
else:
    plt.xlabel("nu", fontsize="large")
plt.ylabel("Jain's Fairness Index", fontsize="large")
# if USETEXLIVE:
#     plt.text(0.5, 0.95, f"$\\alpha={ALPHA}$, $\\nu = {config_dict['FAIRCBFAIRNESS']}$, $\\delta={SMALL_REWARD}$", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
# plt.title("Jain's Fairness Index Plot (Full Information Setting)", fontsize="large")
plt.savefig(VARYING_NUS_PLOT_PATH, bbox_inches='tight', pad_inches=0.01)