"""
AdaBoost classifier provided by Scikit-learn.
"""

import sklearn
from ..node import Node
from sklearn.ensemble import AdaBoostClassifier
from sklearn.exceptions import NotFittedError


class SklAdaBoostClassifier(Node):
    """ xxxx """

    def __init__(self, params=None):
        self.model = AdaBoostClassifier()
        self.params = params
        self.train_index = None
        self.valid_index = None

    def train(self):
        """ ccc """

        # Get X and y values from train_set
        X = self.train_set.drop(self.target, axis=1).values
        y = self.train_set[self.target].values

        # Get only the desired CV indexes
        X_cv_train = X[self.train_index]
        y_cv_train = y[self.train_index]

        self.model.fit(X_cv_train, y_cv_train)

    def predict(self):
        """ xxx """

        try:
            return self.model.predict(self.test_set)
        except NotFittedError as e:
            raise NotFittedError(str(e) + '\n\n' + "sklearn AdaBoost classifier can not predict before fitting")