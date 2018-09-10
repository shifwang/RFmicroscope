from sklearn.ensemble import RandomForestClassifier
from rfms.sklearn_interface import *
import numpy as np
import scipy as sp
def all_tree_paths_test1():
    n_features = 10
    n_samples = 100
    Xtrain = np.random.uniform(0, 1, (n_samples, n_features))
    ytrain = np.random.choice([0, 1], (n_samples, ))
    Xtrain[:,0] = Xtrain[:,0] + ytrain
    rf = RandomForestClassifier(max_features = n_features,
                                n_estimators = 1, bootstrap=False)
    rf.fit(X = Xtrain, y = ytrain)
    tree0 = rf.estimators_[0]
    paths = all_tree_paths(dtree = tree0)
    assert (paths == [[0, 1], [0, 2]]), \
    'paths is supposed to be [[0, 1], [0, 2]] but got %s'%str(paths)
print('running test of sklearn_interface.')
all_tree_paths_test1()
print('passed.')