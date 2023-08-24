from Policy import Policy
import numpy as np
import math

class Hedge(Policy):
    """
    Class representing the classic Hedge policy for the experts problem. This is a
    context-independent policy, and hence will ignore any contexts given to it.

    :param int num_contexts: The number of contexts.
    :param int num_arms: The number of arms.
    :param int T: The time horizon.
    """
    def __init__(self, num_contexts, num_arms, T):
        super().__init__(num_contexts, num_arms)
        self.eta = math.sqrt(math.log(num_arms) / T)
        self.weights = np.ones((self.num_arms, ))

    def decision(self, context=None):
        # ignore context
        return np.random.choice(self.num_arms, p=(self.weights / np.sum(self.weights))) + 1

    def feedback(self, rewards):
        self.weights = self.weights * np.exp(self.eta * rewards)
