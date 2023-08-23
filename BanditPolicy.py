from abc import ABC, abstractmethod

class BanditPolicy(ABC):
    """
    :param int num_contexts: Number of contexts.
    :param int num_arms: Number of arms.
    """
    def __init__(self, num_contexts, num_arms):
        # assume that contexts and arms are indexed from 1
        self.num_contexts = num_contexts
        self.num_arms = num_arms

    """
    :param int context: The current context. Should be in the range [0, num_contexts - 1]
    :returns: The chosen arm in the range [1, num_arms].
    :rtype: int
    """
    @abstractmethod
    def decision(self, context):
        pass

    """
    :param float reward: The reward for the chosen arm.
    """
    @abstractmethod
    def feedback(self, reward):
        pass

