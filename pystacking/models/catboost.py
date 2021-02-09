"""
CatBoost

Further information: https://github.com/catboost/catboost
"""

import catboost as ctb
import pandas as pd
from dmle.stacking.node import Node


class CatBoost(Node):
    """ xxxx """

    def __init__(self):
        """ ccc """
        Node.__init__(self)
        self.model_type = 'catboost'
        self.train_mode = 'cv'
        self.predict_mode = 'cv'

    def train(self, X_train, X_valid, y_train, y_valid):
        """ ccc """

        # Convert data to CatBoost Pool format.
        ds_train = ctb.Pool(X_train, y_train)
        ds_valid = ctb.Pool(X_valid, y_valid)

        # Set context dependent CatBoost parameters.
        self.params['dtrain'] = ds_train
        self.params['eval_set'] = ds_valid

        # Train using parameters sent by the user.
        return ctb.train(**self.params)

    def predict(self, model, test_set):
        """ xxx """

        ds_test = ctb.Pool(test_set)

        # Make predictions using the best training round
        return model.predict(data=ds_test,
                             prediction_type='Probability')[:, 1]

    def feature_importance(self):
        """ Show the feature importance to the entire set of models. """

        fi = pd.DataFrame()

        fi['feature'] = self.models[0].get_feature_importance(prettified=True)['Feature Index']
        fi['importance'] = 0

        for m in self.models:
            fi['importance'] += m.get_feature_importance(prettified=True)['Importances']/len(self.models)

        return fi
