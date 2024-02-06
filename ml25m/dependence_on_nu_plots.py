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
    labels = [f"\\text{{FairCB}}", f"\\text{{This paper with }}$\\alpha = {ALPHA}$"]
else:
    labels = [f"FairCB", f"This paper with alpha = {ALPHA}"]
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams["figure.figsize"] = (5, 4)

plt.figure(0)
ax = plt.axes()
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth(1)  
ax.tick_params(direction='out', length=5, width=0.5, grid_alpha=0.5)
plt.plot(nus, fairness_values, color='b', label=labels[0])
plt.axhline(y=alphaFairCBValue, color='r', label=labels[1])
if USETEXLIVE:
    plt.xlabel(r"$\nu$", fontsize="large")
else:
    plt.xlabel("nu", fontsize="large")
plt.ylabel("Jain's Fairness Index", fontsize="large")
plt.legend(loc="center left", fontsize="large")
# if USETEXLIVE:
#     plt.text(0.5, 0.95, f"$\\alpha={ALPHA}$, $\\nu = {config_dict['FAIRCBFAIRNESS']}$, $\\delta={SMALL_REWARD}$", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
# plt.title("Jain's Fairness Index Plot (Full Information Setting)", fontsize="large")
plt.savefig(VARYING_NUS_PLOT_PATH, bbox_inches='tight', pad_inches=0.01)