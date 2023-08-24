from abc import ABC, abstractmethod

class BanditPolicy(ABC):
    """
    Abstract class representing a bandit policy. All bandit policy classes
    must subclass this.

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
        Return the next arm given the ``context``. Policies which are context-independent can just
        ignore ``context``.

        :param int context: The current context. Should be in the range [0, num_contexts - 1]
        :returns: The chosen arm in the range [1, num_arms].
        :rtype: int
        """
        pass

    @abstractmethod
    def feedback(self, reward):
        """
        Feedback the ``reward`` to the policy for further updates.

        :param float reward: The reward for the arm just picked by the policy.
        """
        pass

