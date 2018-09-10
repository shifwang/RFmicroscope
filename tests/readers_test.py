options = dict()
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from rfms.readers import *

def test1_readers():
    X_train = np.array([[1.3, 0, 0], [1.3, 0, 1], [1.3, 0, 2]])
    y_train = np.array([0, 0, 1])
    rf = RandomForestClassifier(
        n_estimators=1, random_state=1231, max_features = 2)
    rf.fit(X=X_train, y=y_train)
    e0 = rf.estimators_[0]
    a = OrdinaryTreeReader(feature_names = ['f%d'%i for i in range(3)],
    sample_names = ['s%d'%i for i in range(3)])
    a.read_from(e0, X_train)
    assert a.info_.shape == (2, 9), 'a.info_ should have shape (2, 9) but got (%d, %d)'%(a.info_.shape[0], a.info_.shape[1])
    assert tuple(a.info_.loc[0, a.sample_names_]) == (True, True, False), 'a.info_ should only use feature f3 (should see (False, False, True) but got (%r, %r, %r))'%(a.info_.loc[0, 's0'], a.info_.loc[0, 's1'], a.info_.loc[0, 's2'])
    assert tuple(a.info_.loc[0, a.feature_names_]) == (False, False, True), 'a.info_ should only use feature f3 (should see (False, False, True) but got (%r, %r, %r))'%(a.info_.loc[0, 'f0'], a.info_.loc[0, 'f1'], a.info_.loc[0, 'f2'])
    b = ForestReader()
    b.read_from(rf, X_train,TreeReaderType = 'Importance')
    assert b.all_trees_.info_.shape == (2, 9), 'b.all_trees_.info_ should have shape (2, 9) but got (%d, %d)'%(b.all_trees_.info_.shape[0], b.all_trees_.info_.shape[1])
    assert tuple(b.all_trees_.info_.loc[0, b.feature_names_]) == (0, 0, -1/3), 'b.all_trees_.info_ should only use feature f3 (should see (0, 0, -1/3) but got (%f, %f, %f))'%(b.all_trees_.info_.loc[0, 'f0'], b.all_trees_.info_.loc[0, 'f1'], b.all_trees_.info_.loc[0, 'f2'])
def test2_readers():
    X_train = np.array([[1.3, 0, 0], [1.3, 0, 1], [1.3, 0, 2], [1.3, 0, 3]])
    y_train = np.array([0, 0, 1, 0])
    rf = RandomForestClassifier(
        n_estimators=1, random_state=1231, max_features = 2)
    rf.fit(X=X_train, y=y_train)
    e0 = rf.estimators_[0]
    a = OrdinaryTreeReader(feature_names = ['f%d'%i for i in range(3)],
    sample_names = ['s%d'%i for i in range(4)])
    a.read_from(e0, X_train)
    assert a.info_.shape == (3, 10), 'a.info_ should have shape (3, 10) but got (%d, %d)'%(a.info_.shape[0], a.info_.shape[1])
    assert tuple(a.info_.loc[0, a.feature_names_]) == (False, False, True), 'a.info_ should only use feature f3 (should see (False, False, True) but got (%r, %r, %r))'%(a.info_.loc[0, 'f0'], a.info_.loc[0, 'f1'], a.info_.loc[0, 'f2'])
    b = ForestReader()
    b.read_from(rf, X_train,TreeReaderType = 'Importance')
    #b.summary()
    assert b.all_trees_.info_.shape == (3, 10), 'b.all_trees_.info_ should have shape (3, 10) but got (%d, %d)'%(b.all_trees_.info_.shape[0], b.all_trees_.info_.shape[1])
    assert tuple(b.all_trees_.info_.loc[0, b.feature_names_]) == (0, 0, -1/4), 'b.all_trees_.info_ should only use feature f3 (should see (0, 0, -1/3) but got (%f, %f, %f))'%(b.all_trees_.info_.loc[0, 'f0'], b.all_trees_.info_.loc[0, 'f1'], b.all_trees_.info_.loc[0, 'f2'])
print('running test of readers...')
test1_readers()
test2_readers()
print('passed')