from abc import ABC, abstractmethod

class Policy(ABC):
    """
    Abstract class representing a policy for the contextual multi-armed bandits
    problem in the full information setting. All such policies must subclass this class.

    :param int num_contexts: Number of contexts.
    :param int num_arms: Number of arms.
    """
    def __init__(self, num_contexts, num_arms):
        # assume that contexts and arms are indexed from 1
        self.num_contexts = num_contexts
        self.num_arms = num_arms

    @abstractmethod
    def decision(self, context):
        """
        Get the next arm to be played by the policy after observing the ``context``.

        :param int context: The current context. Should be in the range [0, num_contexts - 1]
        :returns: The chosen arm in the range [1, num_arms].
        :rtype: int
        """
        pass

    @abstractmethod
    def feedback(self, rewards):
        """
        Feed the reward vector to the policy for further updates.

        :param numpy.ndarray rewards: A vector representing rewards for each arm.
        """
        pass

