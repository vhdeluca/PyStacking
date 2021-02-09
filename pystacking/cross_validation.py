from sklearn.model_selection import KFold, StratifiedKFold


class CrossValidation(object):

    def __init__(self,
                 dataset=None,
                 n_folds=None,
                 stratified=None):
        self.dataset = dataset
        self.n_folds = n_folds
        self.stratified = stratified
        self.kf = None
        self.folds = None

    def load_ds(self,
                dataset=None,
                n_folds=None,
                stratified=None):
        """ comments here """
        if dataset is not None:
            self.dataset = dataset
        if n_folds is not None:
            self.n_folds = n_folds
        if stratified is not None:
            self.stratified = stratified

        if self.stratified:
            self.kf = StratifiedKFold(n_splits=self.n_folds)
        else:
            self.kf = KFold(n_splits=self.n_folds)

    def get_split(self):
        """ Return the iterable responsible for generating the folds. """
        if self.kf is None:
            self.load_ds()

        if self.stratified:
            return self.kf.split(
                self.dataset.ds.drop([self.dataset.target], axis=1),
                self.dataset.ds[self.dataset.target])
        else:
            return self.kf.split(self.dataset.ds)

    def make_folds(self):
        """ Build all folds. """
        self.folds = list(self.get_split())
