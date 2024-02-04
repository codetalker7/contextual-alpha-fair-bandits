import cvxpy as cp
from tqdm import tqdm
from driver import *

variables = cp.Variable(shape=(NUM_CONTEXTS, NUM_ARMS))
all_ones_arms = np.ones(NUM_ARMS)
all_ones_context = np.ones(NUM_CONTEXTS)

constraints = [
        0 <= variables,
        variables <= 1,
        variables@all_ones_arms == all_ones_context,
]

cumulative_rewards = cp.expressions.constants.Constant(np.ones(NUM_ARMS))
offline_optimal_values = []

for t in tqdm(range(len(data))):
    data_point = data.iloc[t]
    movieId = int(data_point["movieId"])
    context_t = map_user_to_index[int(data_point["userId"])]
    rewards_t = reward_function(movieId)

    cumulative_rewards += rewards_t.T @ variables[context_t]
    objective_function = ((cumulative_rewards ** (1 - ALPHA)) / (1 - ALPHA)).sum()

    obj = cp.Minimize(-objective_function)
    problem = cp.Problem(obj, constraints)
    problem.solve()

    offline_optimal_values.append(-problem.value) 

with open(OFFLINE_OPTIMAL_FILE, "wb") as f:
    pickle.dump(offline_optimal_values, f)
