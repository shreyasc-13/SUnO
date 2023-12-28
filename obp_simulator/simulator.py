import pdb
import numpy as np
from collections import defaultdict
import pickle

import yaml
from munch import munchify

from obp.dataset import (
    logistic_reward_function,
    SyntheticSlateBanditDataset,
)
import warnings
warnings.filterwarnings('ignore')

class OBPSimulator:
    def __init__(self, num_slots, num_actions, sample_size, context_size, reward_structure, reward_type, eval_optimal):
        
        self.num_slots = num_slots
        self.num_actions = num_actions
        self.sample_size = sample_size
        self.context_size = context_size
        self.eval_optimal = eval_optimal
        self.reward_type = reward_type
        self.random_state = 12345
        self.reward_structure = reward_structure

        

    def _define_policies(self):

        # Uniform sampling policy
        self.sampling_policy = None

        # Eval policy defined from the user_action_set
        # Two options to make it optimal and slightly antioptimal, as done
        # here: https://github.com/st-tech/zr-obp/blob/master/examples/quickstart/synthetic_slate.ipynb

        base_expected_reward = self.dataset_with_random_behavior.base_reward_function(
            context=self.bandit_feedback_with_random_behavior["context"],
            action_context=self.dataset_with_random_behavior.action_context,
            random_state=self.dataset_with_random_behavior.random_state,
        )
        if self.eval_optimal:
            self.eval_policy = base_expected_reward * 3
        else:
            self.eval_policy = base_expected_reward * -3
        

    def collect_data(self):

        self.dataset_with_random_behavior = SyntheticSlateBanditDataset(
            n_unique_action=self.num_actions,
            len_list=self.num_slots,
            dim_context=self.context_size,
            reward_type=self.reward_type,
            reward_structure=self.reward_structure,
            click_model=None,
            random_state=self.random_state,
            behavior_policy_function=None,  # set to uniform random
            base_reward_function=logistic_reward_function,
            # is_factorizable = True # Factorized evaluation policy
        )


        self.bandit_feedback_with_random_behavior = self.dataset_with_random_behavior.obtain_batch_bandit_feedback(
            n_rounds=self.sample_size,
            return_pscore_item_position=True,
        )

        random_policy_value = self.dataset_with_random_behavior.calc_on_policy_policy_value(
            reward=self.bandit_feedback_with_random_behavior["reward"],
            slate_id=self.bandit_feedback_with_random_behavior["slate_id"],
        )

        return random_policy_value

    def _compute_y_k(self):
        
        eval_policy_pscores = self.dataset_with_random_behavior.obtain_pscore_given_evaluation_policy_logit(
            action=self.bandit_feedback_with_random_behavior["action"],
            evaluation_policy_logit_=self.eval_policy
        )[1]
        sampling_policy_pscores = self.bandit_feedback_with_random_behavior['pscore_item_position']

        return np.divide(eval_policy_pscores, sampling_policy_pscores)

    def get_weights(self, wts):
        assert (len(wts) == self.num_slots*self.sample_size)
        assert (len(self.bandit_feedback_with_random_behavior['reward']) == self.num_slots*self.sample_size)

        rewards = []
        importance_weights = []
        G_weights = []
        
        for i in range(self.sample_size):
            sample_wts = wts[i*self.num_slots: (i+1)*self.num_slots]
            iw = np.prod(sample_wts)
            g = 1 + np.sum(sample_wts) - self.num_slots
            
            # NOTE: Summing over the slot rewards
            rew = np.sum(self.bandit_feedback_with_random_behavior['reward'][i*self.num_slots: (i+1)*self.num_slots])

            importance_weights.append(iw)
            G_weights.append(g)
            rewards.append(rew)

        return np.array(importance_weights), np.array(G_weights), np.array(rewards)
        

if __name__=='__main__':
    conf = munchify(yaml.safe_load(open('settings.yaml')))
    env = conf.env
    num_slots = env.num_slots
    num_actions = env.num_actions

    alg = conf.alg
    sample_size = alg.sample_size
    num_runs = alg.num_runs
    eval_optimal = alg.eval_optimal # Flag that makes the evaluation policy a little more suboptimal

    obp = conf.obp
    context_size = obp.context_size
    reward_structure = obp.reward_structure
    reward_type = obp.reward_type

    sim = OBPSimulator(num_slots, num_actions, sample_size, context_size, reward_structure, reward_type, eval_optimal)
    sim.collect_data()
    sim._define_policies()
    wts = sim._compute_y_k()
    IW, G, rewards = sim.get_weights(wts)
    
