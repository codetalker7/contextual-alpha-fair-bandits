import numpy as np
import utils

from BanditPolicy import BanditPolicy

class ScaleFreeMAB(BanditPolicy):
    """
    :param int num_contexts: Number of contexts.
    :param int num_arms: Number of arms.
    """
    def __init__(self, num_contexts, num_arms):
        super().__init__(num_contexts, num_arms)
        self.last_eta = num_arms                                # eta for last time stamp
        self.last_gamma = 0.5                                   # gamma for last time stamp
        self.p = np.ones((num_arms, )) / num_arms               # store the last probability vector
        self.sampling_scheme = np.ones((num_arms, ))            # store the last sampling scheme vector
        self.last_chosen_arm = 1                                # the last chosen arm
        self.M_sum = 0                                          # as defined in step 6 of the algorithm
        self.Gamma_sum = 0                                      # as defined in step 7 of the algorithm
        self.cum_estimation_schemes = np.zeros((num_arms, ))    # cumulative sum of estimation schemes

    def decision(self, context):
        # ignore context
        self.sampling_scheme = (1 - self.last_gamma) * self.p + self.last_gamma / self.num_arms
        self.last_chosen_arm = np.random.choice(self.num_arms, p=self.sampling_scheme) + 1
        return self.last_chosen_arm

    def feedback(self, reward):
        # get the characteristic vector of last chosen arm
        arm_characteristic_vector = np.zeros((self.num_arms, ))
        arm_characteristic_vector[self.last_chosen_arm - 1] = 1

        # get the estimation scheme, and update cumulative sum of estimation schemes
        estimation_scheme = (reward / self.sampling_scheme[self.last_chosen_arm - 1]) * arm_characteristic_vector
        self.cum_estimation_schemes += estimation_scheme

        # computing next gamma, and updating Gamma_sum
        self.Gamma_sum += (self.last_gamma * abs(reward))/((1 - self.last_gamma)*self.p[self.last_chosen_arm - 1] + self.last_gamma / self.num_arms) 
        self.last_gamma = self.num_arms / (2 * self.num_arms + self.Gamma_sum)

        # computing next eta, and updating M_sum
        _, opt_val = utils.log_opt(-self.last_eta * estimation_scheme - 1 / self.p)
        self.M_sum += (1/self.last_eta)*opt_val + np.inner(estimation_scheme, self.p) - (1/self.last_eta)*np.log(self.p).sum() + (1 / self.last_eta) * np.inner((1 / self.p), self.p)
        self.last_eta = self.num_arms / (1 + self.M_sum)

        # computing next iterate
        self.p, _ =  utils.log_opt(-self.last_eta * self.cum_estimation_schemes)
