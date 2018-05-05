#!/usr/bin/python

import numpy as np
import sklearn
from sklearn import tree
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier

def all_tree_paths(dtree, root_node_id=0):
    """
    Get all the individual tree paths from root node to the leaves
    for a decision tree classifier object [1]_.

    Parameters
    ----------
    dtree : DecisionTreeClassifier object
        An individual decision tree classifier object generated from a
        fitted RandomForestClassifier object in scikit learn.

    root_node_id : int, optional (default=0)
        The index of the root node of the tree. Should be set as default to
        0 and not changed by the user

    Returns
    -------
    paths : list
        Return a list containing 1d numpy arrays of the node paths
        taken from the root node to the leaf in the decsion tree
        classifier. There is an individual array for each
        leaf node in the decision tree.

    Notes
    -----
        To obtain a deterministic behaviour during fitting,
        ``random_state`` has to be fixed.

    References
    ----------
        .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> raw_data = load_breast_cancer()
    >>> X_train, X_test, y_train, y_test = train_test_split(
        raw_data.data, raw_data.target, train_size=0.9,
        random_state=2017)
    >>> rf = RandomForestClassifier(
        n_estimators=3, random_state=random_state_classifier)
    >>> rf.fit(X=X_train, y=y_train)
    >>> estimator0 = rf.estimators_[0]
    >>> tree_dat0 = all_tree_paths(dtree = estimator0,
                                   root_node_id = 0)
    >>> tree_dat0
    ...                             # doctest: +SKIP
    ...
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
    """
    #TODO: use the decision path function in sklearn to optimize the code
    # sanity CHECK
    if type(dtree) != sklearn.tree.DecisionTreeClassifier:
        raise ValueError('dtree type is supposed to be sklearn.tree.tree.DecisionTreeClassifier but got %s'%type(dtree))

    # Use these lists to parse the tree structure
    children_left = dtree.tree_.children_left
    children_right = dtree.tree_.children_right

    if root_node_id is None:
        paths = []

    if root_node_id == _tree.TREE_LEAF:
        raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

    # if left/right is None we'll get empty list anyway
    if children_left[root_node_id] != _tree.TREE_LEAF:
        paths = [[root_node_id] + l
                 for l in all_tree_paths(dtree, children_left[root_node_id]) +
                 all_tree_paths(dtree, children_right[root_node_id])]

    else:
        paths = [[root_node_id]]
    return paths
