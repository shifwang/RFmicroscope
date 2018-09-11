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
    out,_ = individual_signed_feature_importance(b)
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
    print(rf.feature_importances_)
    out,_ = individual_signed_feature_importance(b)
    assert out.loc['s0', 'f1'] == 1/4, 's0, f1 should be 1/4 got %f'%out.loc['s0', 'f1']
    assert out.loc['s2', 'f1'] == 1/4, 's2, f1 should be 1/4 got %f'%out.loc['s2', 'f1']
def individual_signed_feature_importance_test3():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    raw_data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        raw_data.data, raw_data.target, train_size=0.3,
        random_state=2017)
    #X_train = np.array([[1, -1, 0], [1, 1, 0], [-1, 0, 1], [-1, 0, -1], [-1, 0, -1], [0, 0, 0]])
    #y_train = np.array([1, 0, 1, 0, 0, 1])
    rf = RandomForestClassifier(
        n_estimators=4, random_state=1231, max_features = None, min_impurity_decrease=0.0, min_samples_split = 2, bootstrap=False)
    rf.fit(X=X_train, y=y_train)
    b = ForestReader()
    b.read_from(rf, X_train,TreeReaderType = 'Importance')
    out,to_compare = individual_signed_feature_importance(b, y_train)
    #print(out)
    assert max(abs(rf.feature_importances_ - to_compare)) < 1e-6, 'feature importances not consistent.'
if __name__ == '__main__':
    print('running test of storytellers...')
    individual_signed_feature_importance_test1()
    individual_signed_feature_importance_test2()
    individual_signed_feature_importance_test3()
    print('passed.')
