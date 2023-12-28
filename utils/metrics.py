import numpy as np

class Metrics:
    def __init__(self):
        pass

    def _find_nearest_idx(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def mean(self, cdf, bins):
        mu = 0
        for id in range(1, len(bins[1:])):
            mu += (cdf[id] - cdf[id - 1]) * bins[id]
        return mu

    def alpha_quantile(self, cdf, bins, alpha):
        assert(alpha <= 1 and alpha >= 0)
        idx = self._find_nearest_idx(cdf, alpha)
        return bins[idx+1]

    def cvar(self, cdf, bins, alpha):
        assert(alpha <= 1 and alpha >= 0)
        idx = self._find_nearest_idx(cdf, alpha)
        return self.mean(cdf[:idx+1], bins[:idx+2])/alpha

    def pdf_from_cdf(self, cdf, bins):
        pdf = []
        for id in range(1, len(bins[1:])):
            pdf.append(cdf[id] - cdf[id - 1])
        return np.array(pdf)
    