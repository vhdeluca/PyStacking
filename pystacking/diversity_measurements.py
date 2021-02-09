"""
Diversity measurements.
"""

import pandas as pd
import itertools
import math
import sys


class DiversityMeasurements(object):
    """ Diversity measurements in classifiers. """

    def __init__(self, ds, ground_truth, threshold=None):
        """ ccc """
        self.threshold = threshold
        self.ds = ds
        self.gt = ground_truth

        if self.threshold is None:
            self.threshold = {k: 0.5 for k in ds.columns}

        self.ds_pred = self.get_ds_predictions()
        self.ds_corr = self.get_ds_correct_answers()

    def get_ds_predictions(self):
        """ Generate dataset with label predictions. """

        # Temporary dataframe
        df = pd.DataFrame()

        # Predictions to each column based on threshold.
        for column in self.ds:
            df[column] = self.ds[column].apply(
                lambda x: 1 if x > self.threshold[column] else 0
            )

        return df

    def get_ds_correct_answers(self):
        """ Generate dataset with label predictions. """

        # Temporary dataframe
        df = pd.DataFrame()

        if self.ds_pred.shape[0] != self.gt.shape[0]:
            raise ValueError("Dataset number of rows can not be"
                             "different from ground truth array.")

        # Votes to each column based on threshold and weight
        for column in self.ds_pred:
            df[column] = (self.ds_pred[column] == self.gt).astype(int)

        return df

    def N(self, yi, yk):
        """ Return the number of common correct/incorrect
        answers between two classifiers. """

        yi, yk = self._str2col(yi, yk)

        N00 = sum(((yi == 0) & (yk == 0)))
        N01 = sum(((yi == 0) & (yk == 1)))
        N10 = sum(((yi == 1) & (yk == 0)))
        N11 = sum(((yi == 1) & (yk == 1)))

        return N00, N01, N10, N11

    def Q(self, yi, yk):
        """ Return the Yule’s Q statistic. """

        yi, yk = self._str2col(yi, yk)

        N00, N01, N10, N11 = self.N(yi, yk)

        return ((N11 * N00) - (N01 * N10)) / ((N11 * N00) + (N01 * N10))

    def Q_dict(self):
        """ Return the Yule’s Q statistic
        measure for each classifier pair. """
        return self._dict_pairs(self.Q)

    def Q_report(self):
        """ Return the Yule’s Q statistic
        measure report. """
        return self._report(self.Q_dict)

    def P(self, yi, yk):
        """ Return the correlation coefficient ρ (rho)
        between two classifiers. """

        yi, yk = self._str2col(yi, yk)

        N00, N01, N10, N11 = self.N(yi, yk)

        num = (N11 * N00) - (N01 * N10)
        den = (N11 + N10) * (N01 + N00) * (N11 + N01) * (N10 + N00)
        den = math.sqrt(den)

        return num / den

    def P_dict(self):
        """ Return the correlation coefficient ρ (rho)
        measure for each classifier pair. """
        return self._dict_pairs(self.P)

    def P_report(self):
        """ Return the correlation coefficient ρ (rho)
        measure report. """
        return self._report(self.P_dict)

    def Dis(self, yi, yk):
        """ Return the disagreement measure between two classifiers. """

        yi, yk = self._str2col(yi, yk)

        N00, N01, N10, N11 = self.N(yi, yk)

        return (N01 + N10) / (N00 + N01 + N10 + N11)

    def Dis_dict(self):
        """ Return the disagreement measure key for each classifier pair. """
        return self._dict_pairs(self.Dis)

    def Dis_report(self):
        """ Return the disagreement measure report. """
        return self._report(self.Dis_dict)

    def Df(self, yi, yk):
        """ Return the double-fault measure between two classifiers. """

        yi, yk = self._str2col(yi, yk)

        N00, N01, N10, N11 = self.N(yi, yk)

        return N00 / (N00 + N01 + N10 + N11)

    def Df_dict(self):
        """ Return the double-fault measure key for each classifier pair. """
        return self._dict_pairs(self.Df)

    def Df_report(self):
        """ Return the double-fault measure report. """
        return self._report(self.Df_dict)

    def entropy_E(self):
        """ Return the entropy measure E. """
        L = self.ds_corr.shape[1]
        c = 1 / (L - math.ceil(L/2))
        sum = 0

        for v in self.ds_corr.sum(axis=1):
            sum = sum + (c * min(v, L - v))

        return sum / self.ds_corr.shape[0]

    def kw_variance(self):
        """ Returns the Kohavi-Wolpert variance. """
        L = self.ds_corr.shape[1]
        N = self.ds_corr.shape[0]
        sum = 0

        for v in self.ds_corr.sum(axis=1):
            sum = sum + (v * (L - v))

        return (1 / (N * L * L)) * sum

    def _dict_pairs(self, f):
        """ Return a dictionary where the keys are the pairs and
        values are the results as calculed by f. """
        d = {}

        for pair in self._col_comb():
            d[pair] = f(self.ds_corr[pair[0]],
                        self.ds_corr[pair[1]])

        return d

    def _report(self, f):
        """ Return a dictionary with the report where the keys are the
        metric and and values are the results as calculed by f. """

        sum = 0
        max = 0
        max_pair = None
        min = sys.float_info.max
        min_pair = None
        d = {}

        for key, value in f().items():
            if value > max:
                max = value
                max_pair = key
            if value < min:
                min = value
                min_pair = key
            sum = sum + value

        L = self.ds_corr.shape[1]

        d['avg'] = (2 / (L * (L - 1))) * sum
        d['min'] = min
        d['min_pair'] = min_pair
        d['max'] = max
        d['max_pair'] = max_pair

        return d

    def _str2col(self, yi, yk):
        """ If the arguments are strings, take the
        equivalent column from self.ds_corr. """

        if isinstance(yi, str):
            yi = self.ds_corr[yi]
        if isinstance(yk, str):
            yk = self.ds_corr[yk]

        return yi, yk

    def _col_comb(self):
        """ List with combination between all self.ds_corr columns. """

        return list(itertools.combinations(self.ds_corr.columns, r=2))
