import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict


class Reward:
    def __init__(self, num_arms, num_actions):
        self.num_arms = num_arms
        self.num_actions = num_actions

    def create_spans(self, spacing, temperature):
        # spacing = 1000
        # temperature = 8
        self.spacing = spacing
        self.sigmoids = []
        for i in range(self.num_arms):
            start_idx = 2*i/self.num_arms - 1
            end_idx = 2*(i+1)/self.num_arms - 1
            span = np.linspace(start_idx*self.num_actions, end_idx*self.num_actions, spacing*self.num_actions)
            sigmoid = np.array([1./(1 + np.exp(-temperature*x)) for x in span])
            self.sigmoids.append(sigmoid)

    def psi(self, arm, A_k, nu):
        func = self.sigmoids[arm][A_k*self.spacing:(A_k + 1)*self.spacing]
        func = func * (1./ (max(func) - min(func))) * (1./self.num_arms) 
        func -= min(func)
        assert(func.shape[0] == self.spacing)
        
        idx = int(np.floor(nu * self.spacing))
        return func[idx]

    def reward_cdf(self, A, nu):
        slate_rew = 0
        for i in range(self.num_arms):
            slate_rew += self.psi(i, A[i], nu)
        return slate_rew