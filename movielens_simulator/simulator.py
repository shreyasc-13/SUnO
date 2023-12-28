"""
Interface for the Movielens slate simulator
"""

import numpy as np
from collections import defaultdict
import pickle

import yaml
from munch import munchify

class Simulator:
    def __init__(self, num_slots, num_actions, sample_size, eval_eps, eval_optimal):
        self.num_slots = num_slots
        self.num_actions = num_actions
        self.sample_size = sample_size
        self.eps = eval_eps
        self.eval_optimal = eval_optimal

        self._load_data()
        
    def _load_data(self):

        f = open(f'../data/movielens/user_action_set_historylength.pkl', 'rb')
        self.user_action_set = pickle.load(f)
        f.close()

        self.num_users = len(self.user_action_set.keys())
        self.users = list(self.user_action_set.keys())

    def _define_policies(self):

        # Uniform sampling policy
        # Factored policy that picks each action at each slot uniform randomly
        self.sampling_policy = np.ones(self.num_slots)/self.num_actions

        # Eval policy defined from the user_action_set
        # Two options to make it optimal and slightly suboptimal
        # eval_policy[user_id] -> [a_1, a_2, ... a_K] where the a_i at slot i is picked w.p.
        # 1 - eps * (num_actions - 1) and any other action is picked w.p. eps

        self.eval_policy = defaultdict()
        for i in self.users:
            if self.eval_optimal:
                self.eval_policy[i] = np.array([a[0] for a in self.user_action_set[i][:self.num_slots]])
            else:
                self.eval_policy[i] = np.array(np.random.choice([a[0] for a in self.user_action_set[i]], size=self.num_slots, replace=False))

    def _gt_scores(self, user_id):
        gt_dict =  defaultdict()
        for a in self.user_action_set[user_id]:
            gt_dict[a[0]] = a[1]
        return gt_dict

    def compute_slate_reward(self, ground_truth, pred_slate):
        """
        Function that computes the predicted slate reward by comparing ordering vs ground truth -- nDCG

        Args:
            ground_truth (dict): Ground truth scores -- {action: gt_score}
            pred_slate (array): Predicted slate of size K by any policy

        Returns:
            float: [Expected] slate reward
        """
        DCG = 0
        for i, action in enumerate(pred_slate):
            DCG += ground_truth[action]/np.log2(i+2)
        
        topk_gt = np.sort(np.array(list(ground_truth.values())))[::-1][:len(pred_slate)]
        
        IDCG = np.sum([rel/np.log2(i+2) for i, rel in enumerate(topk_gt)])

        if IDCG == 0:
            raise ValueError("IDCG computed to 0, check preprocessing.")

        nDCG = DCG/IDCG
        slate_reward = nDCG

        # TODO: May want to make this expected reward and sample around this?
        return slate_reward

    def _compute_y_k(self, action, user_id):

        # Return array of Y_k's
        sampling_probs = np.copy(self.sampling_policy)
        eval_policy = self.eval_policy[user_id]
        eval_probs = np.zeros(self.num_slots)
        for i in range(self.num_slots):
            if eval_policy[i] == action[i]:
                eval_probs[i] = 1 - self.eps*(self.num_actions - 1)
            else:
                eval_probs[i] = self.eps
        
        return np.divide(eval_probs, sampling_probs)

    def get_weights(self, data):
        
        rewards = []
        importance_weights = []
        G_weights = []

        for user_id in self.users:
            for d in data[user_id]:
                # eval_policy[user_id]: returns the actions (in order) that eval policy will take
                slate_action = d[1]
                
                y_ks = self._compute_y_k(slate_action, user_id)
                iw = np.prod(y_ks)
                g = 1 + np.sum(y_ks) - self.num_slots

                # debugging
                if iw > 1e5:
                    # Usually when there is a perfect match for the slate_action and eval_policy
                    print(d)
                    print(self.eval_policy[user_id])

                importance_weights.append(iw)
                G_weights.append(g)
                rewards.append(d[2])

        return np.array(importance_weights), np.array(G_weights), np.array(rewards)
    
    def collect_data(self, samples=None):
        
        if samples == None:
            sample_size = self.sample_size
        else:
            sample_size = samples

        data = defaultdict(list)
        eval_rewards = []
        for i in range(sample_size):
            # Sample user context
            user_id = np.random.choice(self.num_users)
            # Get ground truth relevance scores
            gt = self._gt_scores(user_id)
            # Predict user slate (w/ factorized logging policy)
            pred_slate = np.random.choice([a[0] for a in self.user_action_set[user_id]], 5)
            # Get slate reward
            rew = self.compute_slate_reward(gt, pred_slate)
            data[user_id].append((user_id, pred_slate, rew)) # No need to log probs since we know the logging policy

            # Eval policy rewards
            eval_pred_slate = np.zeros(self.num_slots)
            for i in range(self.num_slots):
                if np.random.uniform() <= 1 - self.eps*(self.num_actions):
                    eval_pred_slate[i] = self.eval_policy[user_id][i]
                else:
                    eval_pred_slate[i] =np.random.choice([a[0] for a in self.user_action_set[user_id]])
            eval_rew = self.compute_slate_reward(gt, eval_pred_slate)
            eval_rewards.append(eval_rew)

        return data, eval_rewards


if __name__ == '__main__':
    # Setting run arguments
    conf = munchify(yaml.safe_load(open('settings.yaml')))
    env = conf.env
    num_slots = env.num_slots
    num_actions = env.num_actions

    alg = conf.alg
    sample_size = alg.sample_size
    num_runs = alg.num_runs
    eval_eps = alg.eval_eps
    eval_optimal = alg.eval_optimal # Flag that makes the evaluation policy a little more suboptimal

    # Define the simulator object
    sim = Simulator(num_slots, num_actions, sample_size, eval_eps, eval_optimal)
    sim._define_policies()
    print(sim.collect_data(2))