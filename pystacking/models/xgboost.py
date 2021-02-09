"""
XGBoost (eXtreme Gradient Boosting) is very famous in the ML world.

Further information: https://github.com/dmlc/xgboost
"""

import xgboost as xgb
from dmle.stacking.node import Node


class XGBoost(Node):
    """ xxxx """

    def __init__(self):
        """ ccc """
        Node.__init__(self)
        self.model_type = 'xgboost'
        self.train_mode = 'cv'
        self.predict_mode = 'cv'

    def train(self, X_train, X_valid, y_train, y_valid):
        """ ccc """

        # Convert data to XGBoost DMatrix format.
        dm_train = xgb.DMatrix(X_train, y_train)
        dm_valid = xgb.DMatrix(X_valid, y_valid)

        # Set context dependent XGBoost parameters.
        self.params['dtrain'] = dm_train
        self.params['evals'] = [(dm_train, 'xgb_train'),
                                (dm_valid, 'xgb_valid')]

        # Train using parameters sent by the user.
        return xgb.train(**self.params)

    def predict(self, model, test_set):
        """ xxx """

        dm_test = xgb.DMatrix(test_set)

        # Make predictions using the best training round
        return model.predict(data=dm_test,
                             ntree_limit=model.best_ntree_limit)
