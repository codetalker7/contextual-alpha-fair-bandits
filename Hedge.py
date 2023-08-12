from Policy import Policy
import numpy as np
import math

class Hedge(Policy):
    def __init__(self, num_contexts, num_arms, T):
        super().__init__(num_contexts, num_arms)
        self.eta = math.sqrt(math.log(num_arms) / T)
        self.weights = np.ones((self.num_arms, ))

    """
    :param int context: The current context. Should be in the range [1, num_contexts - 1]
    """
    def decision(self, context=None):
        # ignore context
        return np.random.choice(self.num_arms, p=(self.weights / np.sum(self.weights))) + 1

    def feedback(self, rewards):
        self.weights = self.weights * np.exp(self.eta * rewards)
