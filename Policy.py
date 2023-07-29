from abc import ABC, abstractmethod

class Policy(ABC):
    def __init__(self, num_contexts, num_arms):
        # assume that contexts and arms are indexed from 1
        self.num_contexts = num_contexts
        self.num_arms = num_arms

    @abstractmethod
    def decision(self, context):
        pass

    @abstractmethod
    def feedback(self, rewards):
        pass

