import pandas as pd
import numpy as np
from .cross_validation import CrossValidation
from dask import delayed
from threading import Lock


class Data(object):
    """ This class represents the set "Train plus Test" datasets.
    This "union" is necessary throughout the ensemble. """

    def __init__(self, train_ds=None, test_ds=None):
        """ Train and test datasets. """

        self.pre_train_status = None
        self.pos_train_status = None
        self.pre_pred_status = None
        self.pos_pred_status = None

        if train_ds is None:
            self.train_ds = Dataset()
        else:
            self.train_ds = train_ds

        if test_ds is None:
            self.test_ds = Dataset()
        else:
            self.test_ds = test_ds

    def __hash__(self):
        return hash((self.train_ds, self.test_ds))

    def __eq__(self, other):
        return (self.train_ds, self.test_ds) == (other.train_ds,
                                                 other.test_ds)

    @delayed
    def pre_train(self, *arg):
        """ Define actions to be made before training the nodes that
        have this Data as source. """
        self.train_ds.create_cv()

    @delayed
    def pos_train(self, *arg):
        print("POS_training")

    @delayed
    def pre_pred(self, *arg):
        print("PRE_predict")

    @delayed
    def pos_pred(self, *arg):
        print("POS_predict")


class Dataset(object):
    """ This class represents a dataset. At least
    these requirements are addressed here:

    - Necessary methods to handle its data.
    - Control under multitask access.
    - Management of the target column.
    """

    def __init__(self, dataset=None, target=None):
        """ We prevent the target from being assigned twice. This situation
        may happen when more than one node is assigning data here. """

        self.cv = {}
        self.lock_ds = Lock()
        self.lock_target = Lock()
        self._target_assigned = False

        if dataset is None:
            self.ds = pd.DataFrame()
        else:
            self.ds = dataset

        if target is None:
            self.target = None
        else:
            self.target = target

    def assign(self, column, index, values):
        """ Assign new values to the existing dataset
        according to the desired column and index. """

        # Critical section starts here.
        self.lock_ds.acquire()

        # Resize when index is higher than expected.
        if not set(index).issubset(set(self.ds.index)):
            df = pd.DataFrame(np.nan,
                              index=np.arange(max(index) -
                                              self.ds.shape[0] + 1),
                              columns=[column])

            # Append NaN Dataframe to be able to add values afterwards.
            self.ds = self.ds.append(df, ignore_index=True, sort=False)

        # Insert values at desired index and column.
        self.ds.loc[index, column] = values

        # Leave critical section.
        self.lock_ds.release()

    def assign_target(self, target_values, target_name='target'):
        """ Assign the target values to this dataset. """

        # Critical section starts here.
        self.lock_target.acquire()

        if not self._target_assigned:
            self.assign(target_name,
                        list(range(len(target_values))),
                        target_values)
            self.target = target_name
            self._target_assigned = True

        # Leave critical section.
        self.lock_target.release()

    def get_features_values(self, as_df=False):
        """ Return the independent variables values. """

        self.lock_ds.acquire()

        if self.target is None:
            if as_df:
                values = self.ds
            else:
                values = self.ds.values
        else:
            if as_df:
                values = self.ds.drop(self.target, axis=1)
            else:
                values = self.ds.drop(self.target, axis=1).values

        self.lock_ds.release()

        return values

    def get_target_values(self):
        """ Return the target values. """

        self.lock_ds.acquire()
        values = self.ds[self.target].values
        self.lock_ds.release()

        return values

    def get_ncols(self):
        """ Return the number of columns of the dataset. """
        return self.ds.shape[1]

    def get_nrows(self):
        """ Return the number of rows of the dataset. """
        return self.ds.shape[0]

    def create_cv(self):
        """ Create the CV sets for this dataset based on
        the necessary """

        for k, f in self.cv.items():
            cv = CrossValidation(dataset=self,
                                 n_folds=k,
                                 stratified=True)
            cv.make_folds()
            self.cv[k] = cv
