from rfms.storytellers import individual_signed_feature_importance
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from rfms.readers import ForestReader
def individual_signed_feature_importance_test():
    X_train = np.array([[1.3, 0, 0], [1.3, 0, 1], [1.3, 0, 2], [1.3, 0, 3]])
    y_train = np.array([0, 0, 1, 0])
    rf = RandomForestClassifier(
        n_estimators=1, random_state=1231, max_features = 2)
    rf.fit(X=X_train, y=y_train)
    b = ForestReader()
    b.read_from(rf, X_train,TreeReaderType = 'Importance')
    b.summary()
    out = individual_signed_feature_importance(b)
    print(out)


print('running test of storytellers...')
individual_signed_feature_importance_test()
print('passed.')