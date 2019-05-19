import numpy as np
import scipy as sp
'''
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.fixes import bincount
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.classification import _prf_divide

from unittest import TestCase


def g_mean(y_true, y_pred, labels=None, correction=0.001):
    present_labels = unique_labels(y_true, y_pred)

    if labels is None:
        labels = present_labels
        n_labels = None
    else:
        n_labels = len(labels)
        labels = np.hstack([labels, np.setdiff1d(present_labels, labels,assume_unique=True)])

    le = LabelEncoder()
    le.fit(labels)
    y_true = le.transform(y_true)
    y_pred = le.transform(y_pred)
    sorted_labels = le.classes_

    # labels are now from 0 to len(labels) - 1 -> use bincount
    tp = y_true == y_pred
    tp_bins = y_true[tp]

    if len(tp_bins):
        tp_sum = bincount(tp_bins, weights=None, minlength=len(labels))
    else:
        # Pathological case
        true_sum = tp_sum = np.zeros(len(labels))

    if len(y_true):
        true_sum = bincount(y_true, weights=None, minlength=len(labels))

    # Retain only selected labels
    indices = np.searchsorted(sorted_labels, labels[:n_labels])
    tp_sum = tp_sum[indices]
    true_sum = true_sum[indices]

    recall = _prf_divide(tp_sum, true_sum, "recall", "true", None, "recall")
    recall[recall == 0] = correction

    return sp.stats.mstats.gmean(recall)


class TestEvaluator(TestCase):
    def test_g_mean(self):
        cor = 0.001
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 0, 0]
        self.assertAlmostEqual(g_mean(y_true, y_pred, correction=cor), 1.0, places=10)

        y_true = [0, 0, 0, 0]
        y_pred = [1, 1, 1, 1]
        self.assertAlmostEqual(g_mean(y_true, y_pred, correction=cor), cor, places=10)

        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 0]
        self.assertAlmostEqual(g_mean(y_true, y_pred, correction=cor), 0.5, places=10)

        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 1, 0, 0, 1]
        self.assertAlmostEqual(g_mean(y_true, y_pred, correction=cor), (1*cor*cor)**0.33, places=10)

        y_true = [0, 1, 2, 3, 4, 5]
        y_pred = [0, 1, 2, 3, 4, 5]
        self.assertAlmostEqual(g_mean(y_true, y_pred, correction=cor), 1, places=10)

        y_true = [0, 1, 1, 1, 1, 0]
        y_pred = [0, 0, 1, 1, 1, 1]
        self.assertAlmostEqual(g_mean(y_true, y_pred, correction=cor), (0.5*0.75)**0.5, places=10)

'''