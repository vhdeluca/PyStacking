"""
Scikit-learn Linear SVC.
"""

from sklearn.svm import LinearSVC
from dmle.stacking.node import Node


class SklLSVC(Node):
    """ xxxx """

    def __init__(self):
        """ ccc """
        Node.__init__(self)
        self.model_type = 'skl_lsvc'
        self.train_mode = 'cv'
        self.predict_mode = 'cv'

    def train(self, X_train, X_valid, y_train, y_valid):
        """ ccc """

        clf = LinearSVC(**self.params)

        # Train using parameters sent by the user.
        return clf.fit(X_train, y_train)

    def predict(self, model, test_set):
        """ xxx """

        # Make predictions.
        return model.predict(test_set)
