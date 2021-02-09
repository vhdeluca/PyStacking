from bisect import insort
from dask import delayed
from .models.catboost import CatBoost
from .models.lightgbm import LightGBM
from .models.majority_vote import MajorityVote
from .models.xgboost import XGBoost
from .models.sklearn.ensemble.rf_classifier import SklRandomForestClassifier
from .models.sklearn.linear_model.logistic_regression import SklLogisticRegression
from .models.sklearn.neighbors.kneighbors_classifier import SklKNeighborsClassifier
from .models.sklearn.svm.svc import SklSVC
from .models.sklearn.svm.lsvc import SklLSVC


class SubLayer(object):
    """ xxx """

    def __init__(self, level, sublevel):
        self.level = level
        self.sublevel = sublevel
        self.nodes = []
        self.train_status = None
        self.predict_status = None
        self.connections = []
        self.src_dataset = None
        self.dst_datasets = set()

    def __eq__(self, other):
        return self.sublevel == other.sublevel

    def __lt__(self, other):
        return self.sublevel < other.sublevel

    def insert_node(self, model_type, exec_order=None):
        """ Insert a node into this sublayer.
        All nodes must have different execution orders.
        """

        # Create node according to the desired ML model type.
        node = self.create_node(model_type)

        # Set execution order
        if exec_order is not None:
            node.exec_order = exec_order
        else:
            node.exec_order = self._get_lowest_exec_order()

        # Insert node into nodes list
        if node not in self.nodes:
            insort(self.nodes, node)
        else:
            raise ValueError("There is already a node in this sublayer "
                             "with the same execution order!")

        # Set level and sublevel
        node.level = self.level
        node.sublevel = self.sublevel

        return node

    def create_node(self, model_type):
        """ Create a new node according to the desired model type. """
        if model_type == 'catboost':
            return CatBoost()
        elif model_type == 'lightgbm':
            return LightGBM()
        elif model_type == 'skl_random_forest_classifier':
            return SklRandomForestClassifier()
        elif model_type == 'skl_logistic_regression':
            return SklLogisticRegression()
        elif model_type == 'skl_knn':
            return SklKNeighborsClassifier()
        elif model_type == 'skl_svc':
            return SklSVC()
        elif model_type == 'skl_lsvc':
            return SklLSVC()
        elif model_type == 'xgboost':
            return XGBoost()
        elif model_type == 'majority_vote':
            return MajorityVote()
        else:
            raise ValueError("model not implemented!")

    def _get_lowest_exec_order(self):
        """ Execution order starts from 0 (first to be executed). """
        if self.nodes:
            return self.nodes[-1].exec_order+1
        else:
            return 0

    @delayed
    def train(self, *arg):
        """ Call train for each node in the right order. The elements of
        the array are already in the right order (this is defined when
        building the array) """

        for n in self.nodes:
            # Set training data
            n.src_data = self.src_dataset
            # Set destination datasets
            n.dst_data = self.dst_datasets
            # Define if it is the last sublayer
            if not self.connections:
                n.last_sublayer = True
            # Call training
            n.main_train()

    @delayed
    def predict(self, *arg):
        """ ppp """

        for n in self.nodes:
            n.main_predict()
