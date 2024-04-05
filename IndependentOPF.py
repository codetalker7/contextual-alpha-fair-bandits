from Policy import Policy
from OPF import OPF
import utils
import numpy as np
import math

class IndependentOPF(Policy):
    """
    Class representing independent OPF policies, where the number of policies is
    equal to the number of contexts. In simple words, there is one OPF policy for
    each context (and they are not coupled together).

    :param num_contexts: Number of contexts
    :param num_arms: Number of arms
    :param float alpha: Degree of fairness to be used in the concave fair function.
    """
    def __init__(self, num_contexts, num_arms, alpha):
        super().__init__(num_contexts, num_arms)
        self.alpha = alpha
        self.cumulative_rewards = [np.ones((num_arms, )) for i in range(num_contexts)]  # one reward vector for each context
        self.independent_policies = [OPF(num_contexts, num_arms, alpha) for i in range(num_contexts)]
        self.last_decision = np.empty((self.num_arms, ))
        self.last_context = 0

    def decision(self, context):
        self.last_context = context
        self.last_decision = self.independent_policies[context].next_prediction
        return np.random.choice(self.num_arms, p=self.last_decision) + 1

    def feedback(self, rewards):
        # update cumulative_rewards
        self.cumulative_rewards[self.last_context] += self.last_decision * rewards
        self.independent_policies[self.last_context].feedback(rewards, self.cumulative_rewards[self.last_context])
