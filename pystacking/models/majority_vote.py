"""
Majority vote algorithm
"""

import pandas as pd
from dmle.stacking.node import Node


class MajorityVote(Node):
    """ Majority Vote algorithm. """

    def __init__(self):
        """ ccc """
        Node.__init__(self)
        self.model_type = 'majority_vote'
        self.train_mode = 'no_model'
        self.predict_mode = 'no_model'
        self.threshold = None
        self.weight = None

    def train(self):
        """ Execute training. """

        if None in (self.threshold, self.weight):
            self.set_params()

        X = self.src_data.train_ds.get_features_values(as_df=True)

        return self.majority_vote(X)

    def predict(self):
        """ Execute prediction. """

        if None in (self.threshold, self.weight):
            self.set_params()

        X = self.src_data.test_ds.get_features_values(as_df=True)

        return self.majority_vote(X)

    def majority_vote(self, ds):
        """ Returns the majority vote calculation. """

        # Temporary dataframe
        df = pd.DataFrame()

        # Votes to each column based on threshold and weight
        for column in ds:
            df[column] = ds[column].apply(
                lambda x: self.weight[column] if x > self.threshold[column] else 0
            )

        length = df.shape[1]

        # Sum all votes
        df['sum'] = df.sum(axis=1)

        # Get the final result
        df['result'] = df['sum'].apply(
            lambda x: 1 if x/length > 0.5 else 0
        )

        return df['result'].values

    def set_params(self):
        """ Params configuration. """

        if 'threshold' in self.params:
            self.threshold = self.params['threshold']
        else:
            self.threshold = {k: 0.5 for k in self.src_data.train_ds.ds.columns}

        if 'weight' in self.params:
            self.weight = self.params['weight']
        else:
            self.weight = {k: 1 for k in self.src_data.train_ds.ds.columns}
