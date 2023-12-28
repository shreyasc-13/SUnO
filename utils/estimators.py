import numpy as np


class Estimators:
    def __init__(self, num_bins, normalized):
        self.num_bins = num_bins
        self.normalized = normalized

    def IS(self, IW, rewards):
        reweighed_rewards = []
        for i in range(len(rewards)):
            reweighed_rewards.append(IW[i]*rewards[i])
    
        return np.mean(reweighed_rewards)

    def PICVs(self, G, rewards):
        reweighed_rewards = []
        for i in range(len(rewards)):
            reweighed_rewards.append(G[i]*rewards[i])
    
        return np.mean(reweighed_rewards)

    def UnO(self, IW, rewards):
        bins = np.histogram(rewards, bins=self.num_bins)[1]
        cdf = np.zeros(self.num_bins)
        Z = np.sum(IW)

        for i, b in enumerate(bins[1:]):
            if self.normalized:
                cdf[i] = np.sum(np.array(IW)[np.where(np.array(rewards) < b)[0]])/Z
            else:
                cdf[i] = np.sum(np.array(IW)[np.where(np.array(rewards) < b)[0]])/len(IW)

        cdf[cdf >= 1] = 1
        return cdf, bins

    def SUnO(self, G, rewards):
        bins = np.histogram(rewards, bins=self.num_bins)[1]
        cdf = np.zeros(self.num_bins)
        Z = np.sum(G)

        for i, b in enumerate(bins[1:]):
            if self.normalized:
                cdf[i] = np.sum(np.array(G)[np.where(np.array(rewards) < b)[0]])/Z
            else:
                cdf[i] = np.sum(np.array(G)[np.where(np.array(rewards) < b)[0]])/len(G)

        cdf[cdf >= 1] = 1
        return cdf, bins


if __name__=='__main__':
    conf = munchify(yaml.safe_load(open('settings.yaml')))
    est = conf.estimators
    num_bins = est.num_bins
    normalized = est.normalized

    estimators = Estimators(num_bins, normalized)