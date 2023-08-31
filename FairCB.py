from Policy import Policy
import numpy as np
import math
import cvxpy as cp
import utils

class FairCB(Policy):
    """
    Class representing the Fair CB policy from Chen et. al. 2020's paper
    "Fair Contextual Multi-Armed Bandits: Theory and Experiments".
    """
    def __init__(self, num_contexts, num_arms, fairness, T, context_distribution):
        """
        Initialize the FairCB policy.

        :param int num_contexts: The number of contexts.
        :param int num_arms: The number of arms.
        :param float fairness: The fairness level. Must be in the range (0, 1/num_arms).
        :param int T: The time horizon.
        :param numpy.ndarray context_distribution: The distribution out of which the contexts are generated.
        """
        super(FairCB, self).__init__(num_contexts, num_arms)
        self.fairness = fairness
        self.context_distribution = context_distribution
        self.eta = math.sqrt((num_contexts * math.log(num_arms)) / (T * num_arms))
        self.ps = [np.ones((num_arms, )) for i in range(num_contexts)]
        self.cumulative_loss_estimators = [np.zeros((num_arms, )) for i in range(num_contexts)]
        self.last_context = 0
        self.last_chosen_arm = 1
        self.last_decision = np.empty((self.num_arms, ))    # need to remember last decision to update cumulative rewards

    def decision(self, context):
        # save the context
        self.last_context = context

        # do the FTRL projection step with entropic regularizer
        variable = cp.Variable((self.num_contexts, self.num_arms))
        all_ones = np.ones((self.num_arms, ))

        # all rows should be in the probability simplex
        # and the arm hit rate constraint must be satisfied
        constraints = [0 <= variable, variable <= 1, variable @ all_ones == 1, variable.T @ self.context_distribution >= self.fairness]

        # set the objective function
        objective_function = -(1 / self.eta) * cp.sum(cp.entr(variable))
        for context_id in range(self.num_contexts):
            objective_function += variable[context_id, :] @ self.cumulative_loss_estimators[context_id]

        # solve the problem
        obj = cp.Minimize(objective_function)
        problem = cp.Problem(obj, constraints)
        problem.solve()

        # update ps
        for context_id in range(self.num_contexts):
            self.ps[context_id] = utils.projectOnSimplex(variable.value[context_id, :])

        # return decision for this context
        self.last_decision = self.ps[context]
        self.last_chosen_arm = np.random.choice(self.num_arms, p=self.ps[context]) + 1
        return self.last_chosen_arm

    def feedback(self, rewards):
        # loss estimator
        loss_estimator = np.zeros((self.num_arms, ))
        # 1 - reward will be the loss associated to the arm
        loss_estimator[self.last_chosen_arm - 1] = (1 - rewards[self.last_chosen_arm - 1]) / self.ps[self.last_context][self.last_chosen_arm - 1]
        self.cumulative_loss_estimators[self.last_context] += loss_estimator
