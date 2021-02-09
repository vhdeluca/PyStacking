import numpy as np
from .cross_validation import CrossValidation
from sklearn.metrics import classification_report


class Node(object):
    """ Node is the representation of a ML model plus information
    like params, state, execution order and so on.

    Depending on the stacking operation mode, this node might be sent
    to a machine in the cluster or distributed over many machines.

    When operating in the small data mode, the node will by default be
    sent to any machine in the cluster unless the user specifies the
    desired machine (probably due to needs like GPU, etc).
    """

    def __init__(self, cv_folds=5):
        """ vvv """
        self.level = None
        self.sublevel = None
        self.exec_order = None
        self.cv_folds = cv_folds
        self.last_sublayer = False
        self.params = {}
        self.src_data = None
        self.dst_data = set()
        self.models = []
        self.train_mode = None
        self.predict_mode = None

    def __eq__(self, other):
        return self.exec_order == other.exec_order

    def __lt__(self, other):
        return self.exec_order < other.exec_order

    def main_train(self):
        """ Entry point train for this node. """

        if self.train_mode == 'no_model':
            self.no_model_train()
        elif self.train_mode == 'cv':
            self.oof_train()
        else:
            raise ValueError("train_mode not found!")

    def main_predict(self):
        """ Entry point predict for this node. """

        if self.predict_mode == 'no_model':
            self.no_model_predict()
        elif self.predict_mode == 'cv':
            self.oof_predictions()
        else:
            raise ValueError("predict_mode not found!")

    def no_model_train(self):
        """ No model (static) training.  """

        if not self.last_sublayer:
            result = self.train()
            self.send_train_predictions(self.get_node_id(),
                                        list(range(len(result))),
                                        result)

    def no_model_predict(self):
        """ No model (static) predict. """

        result = self.predict()
        self.send_predictions(self.get_node_id(),
                              list(range(len(result))),
                              result)

    def send_train_predictions(self, id, index, values):
        """ Send train predictions to destination datasets. """

        # Send predictions to all destination datasets.
        for data in self.dst_data:
            # Assign predictions to this dataset.
            data.train_ds.assign(id, index, values)

    def send_predictions(self, id, index, values):
        """ Send predictions to destination datasets. """

        for data in self.dst_data:
            # Assign predictions to this dataset.
            data.test_ds.assign(id, index, values)

    def oof_train(self):
        """ Out-of-fold training. """

        # Get X and y values from train_set.
        X = self.src_data.train_ds.get_features_values()
        y = self.src_data.train_ds.get_target_values()

        # Copy target to the destination datasets.
        if not self.last_sublayer:
            for data in self.dst_data:
                data.train_ds.assign_target(y)

        for train_index, valid_index in (self.src_data
                                             .train_ds
                                             .cv[self.cv_folds]
                                             .folds):
            # Train using the received indexes.
            model = self.train(X[train_index],
                               X[valid_index],
                               y[train_index],
                               y[valid_index])

            # Save model for future use.
            self.models.append(model)

            # Predictions to the OOF set if this is not the last layer.
            if not self.last_sublayer:
                values = self.predict(model, X[valid_index])

                # Send predictions to all destination datasets.
                self.send_train_predictions(self.get_node_id(),
                                            valid_index,
                                            values)

    def oof_predictions(self):
        """ Out-of-fold predictions. """
        test_set = self.src_data.test_ds.ds.values

        # Predictions initially zero
        predictions = np.zeros(self.src_data.test_ds.get_nrows())

        # Predictions for the entire test set
        for m in self.models:
            predictions += self.predict(m, test_set)

        # Divide by the number of folds
        predictions = predictions/5

        self.send_predictions(self.get_node_id(),
                              list(range(len(predictions))),
                              predictions)

    def get_node_id(self):
        """ This is the global identificator for this node.
        It is not a hash to be human-readable. """

        return str(self.level) + "_" + \
            str(self.sublevel) + "_" + \
            str(self.exec_order)

    def classification_report(self,
                              ground_truth,
                              threshold=0.5,
                              output_dict=False):
        """ Return the classification report for a node
        based on the given ground truth. """

        # Get the first destination dataset of this node.
        ds = next(iter(self.dst_data))

        predictions = (ds.test_ds.ds[self.get_node_id()]
                         .apply(lambda x: 1 if x > threshold else 0).values)

        return classification_report(ground_truth,
                                     predictions,
                                     digits=4,
                                     output_dict=output_dict)
