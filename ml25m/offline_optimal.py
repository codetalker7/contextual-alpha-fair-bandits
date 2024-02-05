import cvxpy as cp
from tqdm import tqdm
from driver import *
    
# need one vector variable for each constraint
variables = [cp.Variable(NUM_ARMS) for i in range(NUM_CONTEXTS)]

# constraints
constraints = []

## constraint for simplex
for i in range(NUM_CONTEXTS):
    constraints += [0 <= variables[i], variables[i] <= 1]
    constraints += [cp.sum(variables[i]) == 1]

cumulative_rewards = [cp.expressions.constants.Constant(1) for i in range(NUM_ARMS)]
offline_optimal_values = []

for t in tqdm(range(len(data))):
    data_point = data.iloc[t]
    movieId = int(data_point["movieId"])
    context_t = map_user_to_index[int(data_point["userId"])]
    rewards_t = reward_function(movieId)

    objective_function = cp.expressions.constants.Constant(0)

    for i in range(NUM_ARMS):
        cumulative_rewards[i] += rewards_t[i] * variables[context_t][i]
        objective_function += (cumulative_rewards[i] ** (1 - ALPHA)) / (1 - ALPHA)

    obj = cp.Minimize(-objective_function)
    problem = cp.Problem(obj, constraints)
    problem.solve()

    offline_optimal_values.append(-problem.value)

# saving the offline optimal values
filename = f'pickled_files/offline_optimal_rows={config_dict["ROWS"]}_seed={SEED}_alpha={ALPHA}_smallreward={SMALL_REWARD}_usetimestamps={USETIMESTAMPS}_frequency={config_dict["FREQUENCY"]}_frequencymax={config_dict["FREQUENCY_MAX"]}_highfrequency={config_dict["HIGHFREQUENCY"]}.pickle'
with open(filename, "wb") as f:
    pickle.dump(offline_optimal_values, f)
