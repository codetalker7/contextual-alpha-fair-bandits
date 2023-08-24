from Policy import Policy
import utils
import numpy as np
import math

class OPF(Policy):
    """
    Class representing the Online Proportional Fair (OPF) policy from Sinha et. al. 2023.
    This is a context-independent policy, and hence will ignore any contexts given to it.

    :param num_contexts: Number of contexts.
    :param num_arms: Number of arms.
    :param float alpha: Degree of fairness to be used in the concave fair function.
    """
    def __init__(self, num_contexts, num_arms, alpha):
        super().__init__(num_contexts, num_arms)
        self.next_prediction = np.ones((self.num_arms, )) / self.num_arms   # ignore contexts
        self.cumulative_gradient_norm = 0
        self.alpha = alpha

    def decision(self, context=None): 
        # ignore context
        return np.random.choice(self.num_arms, p=self.next_prediction) + 1

    def feedback(self, rewards, cumulative_rewards):
        """
        Feed the observed rewards to the policy along with the ``cumulative_rewards`` for the policy
        to update itself.

        :param numpy.ndarray rewards: Vector representing rewards for each arm.
        :param numpy.ndarray cumulative_rewards: Vector representing the cumulative rewards accrued till the current time step.
        """
        # compute gradient
        gradient = rewards / np.power(cumulative_rewards, self.alpha)
        self.cumulative_gradient_norm += np.linalg.norm(gradient)
        # D = sqrt(2) for our case
        self.next_prediction = utils.projectOnSimplex(self.next_prediction + gradient / math.sqrt(self.cumulative_gradient_norm))
