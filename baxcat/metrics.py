""" Metrics for use with Engine.eval

Metrics must subclass baxcat.metrics.Metric and must have a __call__ method
that takes two pandas Series.
"""
import numpy as np
from scipy.stats import pearsonr


def confmat(obs, pred):
    """ Calculate the confusion matrix quantities for a binary classifier

    Parameters
    ----------
    obs, inf : np.ndarray
        The observed (`obs`) and predicted (`pred`) data. Must contain only 0
        and 1.
    """
    tp = np.sum(np.logical_and(obs == 1, pred == 1))
    tn = np.sum(np.logical_and(obs == 0, pred == 0))
    fp = np.sum(np.logical_and(obs == 0, pred == 1))
    fn = np.sum(np.logical_and(obs == 1, pred == 0))

    return tp, tn, fp, fn


class Metric(object):
    """ Evaluation metric base class """
    def __call__(self, obs, inf):
        raise NotImplementedError

    @staticmethod
    def convert_to_binary_class(threshold, *args):
        return tuple(np.array(x >= threshold, dtype=float) for x in args)


class SquaredError(Metric):
    """ Sum of squarred (L2) error """
    def __call__(self, obs, inf):
        return np.sum((inf-obs)**2.)


class RelativeError(Metric):
    """ Realtive error """
    def __call__(self, obs, inf):
        if np.any(obs == 0):
            raise ZeroDivisionError('obs cannot contain zeros')
        return np.sum(np.abs(inf/obs - 1.))


class Informedness(Metric):
    """ Informedness: sensitivity + specificity - 1 """
    def __init__(self, threshold=None):
        self._threshold = threshold

    def __call__(self, obs, inf):
        x = obs
        y = inf
        if self._threshold is not None:
            x, y = self.convert_to_binary_class(self._threshold, x, y)

        tp, tn, fp, fn = confmat(x, y)

        return tp/(tp+fn) + tn/(tn+fp) - 1.


class Markedness(Metric):
    """ Markedness: precision + recall - 1 """
    def __init__(self, threshold=None):
        self._threshold = threshold

    def __call__(self, obs, inf):
        x = obs
        y = inf
        if self._threshold is not None:
            x, y = self.convert_to_binary_class(self._threshold, x, y)

        tp, tn, fp, fn = confmat(x, y)

        return tp/(tp+fp) + tn/(tn+fn) - 1.


class Correlation(Metric):
    """ Pearson correlation """
    def __call__(self, obs, inf):
        return pearsonr(obs, inf)[0]
