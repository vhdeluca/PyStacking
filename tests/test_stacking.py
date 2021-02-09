import pytest
from pystacking.stacking import Stacking
from pystacking.layer import Layer
from pystacking.sublayer import SubLayer
from pystacking.node import Node


def test_insert_layers():

    # New empty Stacking
    s = Stacking()

    len_before = len(s.layers)

    # Insert Layers in aleatory order
    s.insert_layer(3)
    s.insert_layer(1)
    s.insert_layer(2)
    s.insert_layer(0)

    len_after = len(s.layers)

    # Check length
    assert len_before == 0
    assert len_after == 4


def test_insert_sublayers():

    # New empty Layer
    l = Layer(0)

    len_before = len(l.sublayers)

    # Insert SubLayers in aleatory order
    l.insert_sublayer(0)
    l.insert_sublayer(1)
    l.insert_sublayer(2)
    l.insert_sublayer(3)

    len_after = len(l.sublayers)

    # Check length
    assert len_before == 0
    assert len_after == 4


def test_insert_nodes():

    # New SubLayer at level 0
    sl = SubLayer(0, 0)

    len_before = len(sl.nodes)

    # Insert SubLayers in aleatory order
    sl.insert_node('lightgbm', 1)
    sl.insert_node('lightgbm', 0)
    sl.insert_node('xgboost', 3)
    sl.insert_node('xgboost')

    len_after = len(sl.nodes)

    # Check length
    assert len_before == 0
    assert len_after == 4
    # Check SubLayers order
    assert sl.nodes[0].exec_order == 0
    assert sl.nodes[0].model_type == 'lightgbm'
    assert sl.nodes[1].exec_order == 1
    assert sl.nodes[1].model_type == 'lightgbm'
    assert sl.nodes[2].exec_order == 3
    assert sl.nodes[2].model_type == 'xgboost'
    assert sl.nodes[3].exec_order == 4
    assert sl.nodes[3].model_type == 'xgboost'

    # Check exception inserting the same SubLayers
    with pytest.raises(Exception):
        sl.insert_node('lightgbm', 1)


def test_ensemble_insert_nodes():
    """ In this test, we try a complex combination of node
    insertions and check if they are placed at the right place.
    """
    s = Stacking()

    n3 = s.insert_node('xgboost')
    n3.cv_folds = 1
    n4 = s.insert_node('xgboost')
    n4.cv_folds = 2
    n5 = s.insert_node('xgboost', 1)
    n5.cv_folds = 3
    n6 = s.insert_node('xgboost', 1)
    n6.cv_folds = 4
    n2 = s.insert_node('xgboost', 2, 0)
    n2.cv_folds = 5
    n1 = s.insert_node('xgboost', 2, 1)
    n1.cv_folds = 6
    n7 = s.insert_node('xgboost', 2, 1)
    n7.cv_folds = 7

    assert s.layers[0].sublayers[0].nodes[0].cv_folds == 1
    assert s.layers[0].sublayers[1].nodes[0].cv_folds == 2
    assert s.layers[1].sublayers[0].nodes[0].cv_folds == 3
    assert s.layers[1].sublayers[1].nodes[0].cv_folds == 4
    assert s.layers[2].sublayers[0].nodes[0].cv_folds == 5
    assert s.layers[2].sublayers[1].nodes[0].cv_folds == 6
    assert s.layers[2].sublayers[1].nodes[1].cv_folds == 7
