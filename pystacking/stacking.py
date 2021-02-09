from bisect import insort
from .dataset import Data, Dataset
from .layer import Layer
from .utils import get_key, ds_exec_order


class Stacking(object):
    """ This class represents the whole stacking.

    It contains the following components:
    * Layers: each one has a unique level and contains Sublayers.
    * Datasets: responsible for storing the datasets used throughout
      the stacking.
    """

    def __init__(self):
        self.layers = []
        self.datasets = {}
        self.input_data = {}
        self.output_data = Data()

    def insert_layer(self, layer_level):
        """ Insert one layer into the stacking. It is not possible
        to have more than one layer at the same level. This could be
        achieved by inserting sublayers.
        """

        layer = Layer(layer_level)

        if layer not in self.layers:
            insort(self.layers, layer)
        else:
            raise ValueError("A layer already exists at this level.")

    def insert_dataset(self, train_dataset, target, test_dataset, sublayers=None):
        """ Insert dataset into the input data. """

        ds = Data(Dataset(train_dataset, target), Dataset(test_dataset))

        if isinstance(sublayers, list):
            self.input_data[ds] = sublayers
        else:
            if sublayers is None:
                self.input_data[ds] = None
            else:
                self.input_data[ds] = [sublayers]

    def insert_connection(self, source, destination):
        """ Connections are always a link between a source and a
        destination of data. Source and destination are identified by:
        (layer, sublayer)
        """

        # Connections are two dimension lists
        assert len(source) == 2
        assert len(destination) == 2

        # Destination must be at a higher layer
        assert source[0] < destination[0]

        # Source and destination layers must exist
        assert source[0] < len(self.layers)
        assert destination[0] < len(self.layers)

        # Sublayers must exist at their layers
        assert source[1] < len(self.layers[source[0]].sublayers)
        assert destination[1] < len(self.layers[destination[0]].sublayers)

        self.layers[source[0]]\
            .sublayers[source[1]]\
            .connections\
            .append(destination)

    def _create_connections(self):
        """ If any connection is provided, then this method is called.
        It creates connections between all sublayers from one layer to
        the next one. """

        for l in self.layers:
            for sl in l.sublayers:
                # Check if it's not the last layer
                if (l.level + 1) < len(self.layers):
                    for d_sl in self.layers[l.level + 1].sublayers:
                        self.insert_connection((l.level, sl.sublevel),
                                               ((l.level + 1), d_sl.sublevel))

    def _create_datasets(self):
        """ Create all intermediate datasets necessary to the stacking.
        It avoids creating redundant datasets when different sublayers
        receive data from the same group of sublayers. """

        datasets = {}

        for l in self.layers:
            for s in l.sublayers:
                for c in s.connections:
                    if c in datasets:
                        datasets[c].append((l.level, s.sublevel))
                    else:
                        datasets[c] = [(l.level, s.sublevel)]

        set_datasets = set()

        for key, value in datasets.items():
            set_datasets.add(tuple(value))

        for ds in set_datasets:
            self.datasets[ds] = Data()

        for key, value in datasets.items():
            self.layers[key[0]]\
                .sublayers[key[1]]\
                .src_dataset = self.datasets[tuple(value)]

        for s in self.sublayers():
            # Fill destination datasets
            for c in s.connections:
                s.dst_datasets.add(self.layers[c[0]]
                                       .sublayers[c[1]]
                                       .src_dataset)
            # Fill sublayers without destination dataset
            if not s.connections:
                s.dst_datasets.add(self.output_data)

        # Fill input data
        for data, sublayers in self.input_data.items():
            for sl in sublayers:
                self.layers[sl[0]].sublayers[sl[1]].src_dataset = data


    def insert_node(self, model_type, level=0, sublevel=None):
        """ Insert a node in the stacking.

        Extended description of function.

        Parameters
        ----------
        model_type : str
            Model type to be executed by this node.
        level : int
            Level where the model will be inserted.
        sublevel : int
            Sublevel where the model will be inserted.

        Returns
        -------
        Object
            The generated node object.

        Examples
        --------
        >>> e = Stacking()
        >>> e.insert_node("xgboost")
        >>> e.insert_node("xgboost", 1)
        >>> e.insert_node("xgboost", 2, 0)
        """
        # Check whether layer level exists or not
        if level in [l.level for l in self.layers]:
            # Check whether sublayer sublevel exists or not
            if sublevel not in [sl.sublevel for sl in self.layers[level].sublayers]:
                sub = self.layers[level].insert_sublayer(sublevel)
        else:
            # Necessary to create both layer and sublayer
            self.insert_layer(level)
            sub = self.layers[level].insert_sublayer(sublevel)

        if sublevel is None:
            sublevel = sub.sublevel

        return self.layers[level].sublayers[sublevel].insert_node(model_type)


    def _config_input_data(self):
        """ Configure input data with the right sublayers """
        for data, sublayers in self.input_data.items():
            if sublayers is None:
                self.input_data[data] = [(0, sl.sublevel) 
                                         for sl in self.layers[0].sublayers]


    def _startup_tasks(self):
        """ Perform all necessary startup tasks before training. """
        # Configure Input Data
        self._config_input_data()

        # We consider layer and sublayer 0 to indicate whether
        # connections were made or not.
        if not self.layers[0].sublayers[0].connections:
            self._create_connections()

        if not self.datasets:
            self._create_datasets()

        # Configure Cross Validation
        self._configure_cv()


    def train(self):
        """ Call train following the datasets sequence. """

        # Perform all necessary startup tasks
        self._startup_tasks()

        # Input datasets.
        # The execution order is not important here.
        for data, sublayer in self.input_data.items():
            data.pre_train_status = data.pre_train()
            train_status = []
            for sl in self.sublayers():
                if sl.src_dataset in [self.get_sublayer(x).src_dataset for x in sublayer]:
                    sl.train_status = sl.train(data.pre_train_status)
                    train_status.append(sl.train_status)
            data.pos_train_status = data.pos_train(*train_status)

        # Intermediate datasets.
        # The execution order matters here.
        for ds in ds_exec_order(self.datasets):
            # List of all datasets connected to this one.
            pos_train_status = []
            for sl in ds:
                pos_train_status.append(self.layers[sl[0]]
                                            .sublayers[sl[1]]
                                            .src_dataset.pos_train_status)
            self.datasets[ds].pre_train_status = self.datasets[ds].pre_train(*pos_train_status)
            train_status = []
            for sl in self.sublayers():
                if sl.src_dataset == self.datasets[ds]:
                    sl.train_status = sl.train(self.datasets[ds].pre_train_status)
                    train_status.append(sl.train_status)
            self.datasets[ds].pos_train_status = self.datasets[ds].pos_train(*train_status)

        # Output datasets.
        for sl in self.sublayers():
            pos_train_status = []
            if self.output_data in sl.dst_datasets:
                pos_train_status.append(sl.src_dataset.pos_train_status)
            self.output_data.pre_train_status = self.output_data.pre_train(*pos_train_status)

        # Finally we compute the whole training.
        self.output_data.pre_train_status.compute()


    def _configure_cv(self):
        """ Each dataset contains a set of nodes by which it is the
        source of data. Since each node has a number of folds specified
        for CV, this function populates the key "cv" for each number of
        folds. The value is None here, but will be changed by the object
        when it is created. """

        # Configuring CV
        for sl in self.sublayers():
            # Stacking Data
            for key, ds in self.datasets.items():
                if sl.src_dataset == ds:
                    for n in sl.nodes:
                        ds.train_ds.cv[n.cv_folds] = None
            # Input Data
            for key, ds in self.input_data.items():
                if sl.src_dataset == key:
                    for n in sl.nodes:
                        key.train_ds.cv[n.cv_folds] = None

    def train_exec_graph(self):
        """ Return the train execution graph. """
        return self.output_data.pre_train_status


    def predict_exec_graph(self):
        """ Return the predict execution graph. """
        return self.output_data.pre_pred_status


    def sublayers(self):
        """ List of all sublayers contained in this stacking. """
        return [sublayer for l in self.layers for sublayer in l.sublayers]


    def get_sublayer(self, sl):
        """ Return the sublayer object given the tuple (layer, sublayer). """
        return self.layers[sl[0]].sublayers[sl[1]]


    def predict(self):
        """ Return the predictions to the whole stacking. """

        # Input datasets.
        # The execution order is not important here.
        for data, sublayer in self.input_data.items():
            data.pre_pred_status = data.pre_pred()
            pred_status = []
            for sl in self.sublayers():
                if sl.src_dataset in [self.get_sublayer(x).src_dataset for x in sublayer]:
                    sl.pred_status = sl.predict(data.pre_pred_status)
                    pred_status.append(sl.pred_status)
            data.pos_pred_status = data.pos_pred(*pred_status)

        # Intermediate datasets.
        # The execution order matters here.
        for ds in ds_exec_order(self.datasets):
            # List of all datasets connected to this one.
            pos_pred_status = []
            for sl in ds:
                pos_pred_status.append(self.layers[sl[0]]
                                           .sublayers[sl[1]]
                                           .src_dataset.pos_pred_status)
            self.datasets[ds].pre_pred_status = self.datasets[ds].pre_pred(*pos_pred_status)
            predict_status = []
            for sl in self.sublayers():
                if sl.src_dataset == self.datasets[ds]:
                    sl.predict_status = sl.predict(self.datasets[ds].pre_pred_status)
                    predict_status.append(sl.predict_status)
            self.datasets[ds].pos_pred_status = self.datasets[ds].pos_pred(*predict_status)

        # Output datasets.
        for sl in self.sublayers():
            pos_pred_status = []
            if self.output_data in sl.dst_datasets:
                pos_pred_status.append(sl.src_dataset.pos_pred_status)
            self.output_data.pre_pred_status = self.output_data.pre_pred(*pos_pred_status)

        # Finally we compute the whole prediction.
        self.output_data.pre_pred_status.compute()

        # Return the answers.
        return self.output_data.test_ds.ds


    def node_classification_report(self, node, ground_truth, threshold=0.5):
        """ Return the classification report for a node based on the
        given ground truth. """

        # Get sublayer.
        sl = self.get_sublayer((node[0], node[1]))

        # Call classification report for this node.
        return sl.nodes[node[2]].classification_report(ground_truth,
                                                       threshold=threshold)
