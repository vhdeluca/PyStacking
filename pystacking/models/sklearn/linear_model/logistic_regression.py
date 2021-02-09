"""
Scikit-learn Logistic Regression.
"""

from sklearn.linear_model import LogisticRegression
from dmle.stacking.node import Node


class SklLogisticRegression(Node):
    """ xxxx """

    def __init__(self):
        """ ccc """
        Node.__init__(self)
        self.model_type = 'skl_logistic_regression'
        self.train_mode = 'cv'
        self.predict_mode = 'cv'

    def train(self, X_train, X_valid, y_train, y_valid):
        """ ccc """

        clf = LogisticRegression(**self.params)

        # Train using parameters sent by the user.
        return clf.fit(X_train, y_train)

    def predict(self, model, test_set):
        """ xxx """

        # Make predictions.
        return model.predict_proba(test_set)[:, 1]
