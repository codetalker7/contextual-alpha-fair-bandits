from Policy import Policy
from OPF import OPF
import numpy as np
import math

class ParallelOPF(Policy):
    def __init__(self, num_contexts, num_arms, alpha):
        super().__init__(num_contexts, num_arms)
        self.alpha = alpha
        self.cumulative_rewards = np.ones((num_arms, ))
        self.parallel_policies = [OPF(num_contexts, num_arms, alpha) for i in range(num_contexts)]
        self.last_decision = np.empty((self.num_arms, ))    # need to remember last decision to update cumulative rewards
        self.last_context = 0

    def decision(self, context):
        self.last_context = context
        self.last_decision = self.parallel_policies[context].decision()
        return self.last_decision

    def feedback(self, rewards):
        # update cumulative_rewards
        self.cumulative_rewards = self.cumulative_rewards + self.last_decision * rewards
        self.parallel_policies[self.last_context].feedback(rewards, self.cumulative_rewards)
