"""
LightGBM

Further information: https://github.com/microsoft/LightGBM
"""

import lightgbm as lgb
from dmle.stacking.node import Node


class LightGBM(Node):
    """ xxxx """

    def __init__(self):
        """ ccc """
        Node.__init__(self)
        self.model_type = 'lightgbm'
        self.train_mode = 'cv'
        self.predict_mode = 'cv'

    def train(self, X_train, X_valid, y_train, y_valid):
        """ ccc """

        # Convert data to XGBoost DMatrix format.
        ds_train = lgb.Dataset(X_train, y_train)
        ds_valid = lgb.Dataset(X_valid, y_valid)

        # Set context dependent XGBoost parameters.
        self.params['train_set'] = ds_train
        self.params['valid_sets'] = [ds_valid]
        self.params['valid_names'] = ['lgb_valid']

        # Train using parameters sent by the user.
        return lgb.train(**self.params)

    def predict(self, model, test_set):
        """ xxx """

        # Make predictions using the best training round
        return model.predict(data=test_set,
                             num_iteration=model.best_iteration)
