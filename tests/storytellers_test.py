from rfms.storytellers import individual_signed_feature_importance
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from rfms.readers import ForestReader
def individual_signed_feature_importance_test1():
    X_train = np.array([[1.3, 0, 0], [1.3, 0, 1], [1.3, 0, 2], [1.3, 0, 3]])
    y_train = np.array([0, 0, 1, 0])
    rf = RandomForestClassifier(
        n_estimators=1, random_state=1231, max_features = 2, bootstrap=False)
    rf.fit(X=X_train, y=y_train)
    b = ForestReader()
    b.read_from(rf, X_train,TreeReaderType = 'Importance')
    out = individual_signed_feature_importance(b)
    assert out.loc['s0', 'f2'] == -1/4, 's0, f2 should be -1/4 got %f'%out.loc['s0', 'f2']
    assert out.loc['s2', 'f2'] == 3/4, 's2, f2 should be 3/4 got %f'%out.loc['s2', 'f2']
def individual_signed_feature_importance_test2():
    X_train = np.array([[1, -1], [1, 1], [-1, 1], [-1, -1]])
    y_train = np.array([1, 0, 1, 0])
    rf = RandomForestClassifier(
        n_estimators=2, random_state=1231, max_features = 2, min_impurity_decrease=0.0, min_samples_split = 2, bootstrap=False)
    rf.fit(X=X_train, y=y_train)
    b = ForestReader()
    b.read_from(rf, X_train,TreeReaderType = 'Importance')
    out = individual_signed_feature_importance(b)
    assert out.loc['s0', 'f1'] == 1/4, 's0, f1 should be 1/4 got %f'%out.loc['s0', 'f1']
    assert out.loc['s2', 'f1'] == 1/4, 's2, f1 should be 1/4 got %f'%out.loc['s2', 'f1']
    
if __name__ == '__main__':
    print('running test of storytellers...')
    individual_signed_feature_importance_test1()
    individual_signed_feature_importance_test2()
    print('passed.')