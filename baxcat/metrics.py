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

    Returns
    -------
    tp : int
        The number of true positives
    tn : int
        The number of true negatives
    fp : int
        The number of false positives
    fn : int
        The number of false negatives
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
    """ Sum of squarred (L2) error

    The suqared error of predictions y given observations x is

    .. math:: \sum_{i=1}^n (x_i - y_i)^2
    """
    def __call__(self, obs, inf):
        """
        Examples
        --------

        >>> import pandas as pd
        >>> pred = pd.Series([0.1, 1.2, .8, .7, .2])
        >>> obs = pd.Series([0.25, 1.1, .3, .6, .4])
        >>> sse = SquaredError()
        >>> sse(obs, pred)
        0.3325
        """
        return np.sum((inf-obs)**2.)


class RelativeError(Metric):
    """ Realtive error

    The relative error of predictions y given observations x is

    .. math:: \sum_{i=1}^n |y/x - 1|
    """
    def __call__(self, obs, inf):
        """
        Examples
        --------

        >>> import pandas as pd
        >>> pred = pd.Series([0.1, 1.2, .8, .7, .2])
        >>> obs = pd.Series([0.25, 1.1, .3, .6, .4])
        >>> re = RelativeError()
        >>> re(obs, pred)
        """
        if np.any(obs == 0):
            raise ZeroDivisionError('obs cannot contain zeros')
        return np.sum(np.abs(inf/obs - 1.))


class Accuracy(Metric):
    """ Accuracy (proportion correctly classified) """
    def __call__(self, obs, inf):
        """
        Examples
        --------

        >>> import pandas as pd
        >>> pred = pd.Series([0, 1, 1, 0, 1])
        >>> obs = pd.Series([0, 1, 0, 1, 1])
        >>> acc = Accuracy()
        >>> acc(obs, pred)
        0.6
        """
        return np.sum(obs == inf)/float(len(obs))


class Informedness(Metric):
    """ Informedness: sensitivity + specificity - 1 """
    def __init__(self, threshold=None):
        self._threshold = threshold

    def __call__(self, obs, inf):
        """
        Examples
        --------
        Binary data do not require a threshold

        >>> import pandas as pd
        >>> pred = pd.Series([0, 1, 1, 0, 1])
        >>> obs = pd.Series([0, 1, 0, 1, 1])
        >>> infd = Informedness()
        >>> infd(obs, pred)
        4.666666666666667

        Continuous data are binarized with the threshold. Datum larger than
        `threshold` are assigned 1; values less than `threshold` are assigned 0.

        >>> pred = pd.Series([0.1, 1.2, .8, .7, .2])
        >>> obs = pd.Series([0.25, 1.1, .3, .6, .4])
        >>> infd = Informedness(threshold=0.35)
        >>> infd(obs, pred)
        0.75
        """
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
        """
        Examples
        --------
        Binary data do not require a threshold

        >>> import pandas as pd
        >>> pred = pd.Series([0, 1, 1, 0, 1])
        >>> obs = pd.Series([0, 1, 0, 1, 1])
        >>> mkd = Markedness()
        >>> mkd(obs, pred)
        4.666666666666667

        Continuous data are binarized with the threshold. Datum larger than
        `threshold` are assigned 1; values less than `threshold` are assigned 0.

        >>> pred = pd.Series([0.1, 1.2, .8, .7, .2])
        >>> obs = pd.Series([0.25, 1.1, .3, .6, .4])
        >>> mkd = Markedness(threshold=0.35)
        >>> mkd(obs, pred)
        0.5
        """
        x = obs
        y = inf
        if self._threshold is not None:
            x, y = self.convert_to_binary_class(self._threshold, x, y)

        tp, tn, fp, fn = confmat(x, y)

        return tp/(tp+fp) + tn/(tn+fn) - 1.


class Correlation(Metric):
    """ Pearson correlation """
    def __call__(self, obs, inf):
        """
        Examples
        --------

        >>> import pandas as pd
        >>> obs = pd.Series([1., 2., 1., 2., 3.])
        >>> corr = Correlation()
        >>> corr(obs, obs)
        1.0
        """
        return pearsonr(obs, inf)[0]
