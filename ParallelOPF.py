from Policy import Policy
from OPF import OPF
import numpy as np
import math

class ParallelOPF(Policy):
    """
    :param num_contexts: Number of contexts.
    :param num_arms: Number of arms.
    :param alpha: Degree of fairness to be used in the concave fair function.
    """
    def __init__(self, num_contexts, num_arms, alpha):
        super().__init__(num_contexts, num_arms)
        self.alpha = alpha
        self.cumulative_rewards = np.ones((num_arms, ))
        self.parallel_policies = [OPF(num_contexts, num_arms, alpha) for i in range(num_contexts)]
        self.last_decision = np.empty((self.num_arms, ))    # need to remember last decision to update cumulative rewards
        self.last_context = 0

    """
    :param int context: The current context. Should be in the range [0, num_contexts - 1].
    """
    def decision(self, context):
        self.last_context = context
        self.last_decision = self.parallel_policies[context].next_prediction
        return np.random.choice(self.num_arms, p=self.last_decision) + 1

    def feedback(self, rewards):
        # update cumulative_rewards
        self.cumulative_rewards = self.cumulative_rewards + self.last_decision * rewards
        self.parallel_policies[self.last_context].feedback(rewards, self.cumulative_rewards)
