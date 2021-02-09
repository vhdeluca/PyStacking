"""
Scikit-learn Random Forest Classifier.

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""

from sklearn.ensemble import RandomForestClassifier
from dmle.stacking.node import Node


class SklRandomForestClassifier(Node):
    """ xxxx """

    def __init__(self):
        """ ccc """
        Node.__init__(self)
        self.model_type = 'skl_random_forest_classifier'
        self.train_mode = 'cv'
        self.predict_mode = 'cv'

    def train(self, X_train, X_valid, y_train, y_valid):
        """ ccc """

        clf = RandomForestClassifier(**self.params)

        # Train using parameters sent by the user.
        return clf.fit(X_train, y_train)

    def predict(self, model, test_set):
        """ xxx """

        # Make predictions.
        return model.predict_proba(test_set)[:, 1]
