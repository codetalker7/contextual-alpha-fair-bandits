from Policy import Policy
import utils
import numpy as np
import math

class OPF(Policy):
    def __init__(self, num_contexts, num_arms, alpha):
        super().__init__(num_contexts, num_arms)
        self.next_prediction = np.ones((self.num_arms, ))   # ignore contexts
        self.cumulative_gradient_norm = 0
        self.alpha = alpha

    def decision(self, context=None): 
        # ignore context
        return self.next_prediction

    def feedback(self, rewards, cumulative_rewards):
        # compute gradient
        gradient = rewards / np.power(cumulative_rewards, self.alpha)
        self.cumulative_gradient_norm += np.linalg.norm(gradient)
        # D = sqrt(2) for our case
        self.next_prediction = utils.projectOnSimplex(self.next_prediction + gradient / math.sqrt(self.cumulative_gradient_norm))
