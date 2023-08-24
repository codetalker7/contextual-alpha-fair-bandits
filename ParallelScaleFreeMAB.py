import numpy as np

from BanditPolicy import BanditPolicy
from ScaleFreeMAB import ScaleFreeMAB

class ParallelScaleFreeMAB(BanditPolicy):
    """
    Chaudhary et. al. 2023's fair contextual bandit policy for the bandit setting.

    :param int num_contexts: Number of contexts.
    :param int num_arms: Number of arms.
    :param float alpha: Degree of fairness to be used in the concave fair function.
    """
    def __init__(self, num_contexts, num_arms, alpha):
        super().__init__(num_contexts, num_arms)
        self.alpha = alpha
        self.cumulative_rewards = np.ones((num_arms, ))
        self.parallel_bandit_policies = [ScaleFreeMAB(num_contexts, num_arms) for i in range(num_contexts)]
        self.last_chosen_arm = 1
        self.last_context = 0

    def decision(self, context):
        self.last_context = context
        self.last_chosen_arm = self.parallel_bandit_policies[context].decision(context)
        return self.last_chosen_arm

    def feedback(self, reward):
        # feedback modified reward to underlying policy
        modified_reward = (1 / (self.cumulative_rewards[self.last_chosen_arm - 1]) ** self.alpha) * reward

        self.parallel_bandit_policies[self.last_context].feedback(modified_reward)

        # update cumulative reward for picked arm
        self.cumulative_rewards[self.last_chosen_arm - 1] += reward

