from rfms.readers import *
import numpy as np
import pandas as pd
def individual_signed_feature_importance(forestReader, labels = None):
    '''Compute the feature importance for each sample
    Args:
        forestReader - ImportanceTreeReader, the forestReader learned
        labels       - numpy array, the label of samples fed into forestReader, default None, only used to calculate the feature importance
    Returns:
        out - dataframe, sample * feature
        feature_importance - numpy 1d array, estimated feature importance
    Example:
    >>> import RFmicroscope
    >>> #TODO
    '''
    if forestReader.TreeReaderType != 'Importance':
        raise AssertionError('TreeReaderType must be Importance but got %s'%forestReader.TreeReaderType)
    # initialize a dataframe
    out = {'sample_names': forestReader.sample_names_}
    for f in forestReader.feature_names_:
        out[f] = np.zeros((forestReader.number_of_samples_, ))
    out = pd.DataFrame(out)
    out = out.set_index('sample_names')
    # loop through each leaf, 
    #   for any sample it contains, add the feature importance
    for ind in forestReader.all_trees_.info_.index: # TODO: make sure it is each row
        to_add = forestReader.all_trees_.info_.loc[ind, forestReader.feature_names_]
        for sample in out.index:
            if forestReader.all_trees_.info_.loc[ind,sample]:
                out.loc[sample,:] += to_add
    number_of_trees_ = len(np.unique(forestReader.all_trees_.info_['tree_id']))
    for sample in out.index:
        out.loc[sample,:] = out.loc[sample,:] / number_of_trees_
    if labels is not None:
        if set(labels[:]) != set([0, 1]):
            raise ValueError('only supported 0-1 label right now.')
        feature_importance = 2. * (labels - .5).reshape((1, len(labels))).dot(np.array(out))[0] / len(labels)
        feature_importance = feature_importance / sum(feature_importance)
    else:
        feature_importance = None
    return out, feature_importance
if __name__ == '__main__':
    options = dict()
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    raw_data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        raw_data.data, raw_data.target, train_size=0.5,
        random_state=2017)
    rf = RandomForestClassifier(
        n_estimators=1, random_state=1231, max_depth = 2, bootstrap=True)
    rf.fit(X=X_train, y=y_train)
    #print(rf.estimators_[0].tree_.value[0])
    b = ForestReader()
    b.read_from(rf, X_test, TreeReaderType = 'Importance')
    b.summary()
    out, feature_importances_ = individual_signed_feature_importance(b, y_test)
    print(out.head())
    print(rf.feature_importances_)
    print(feature_importances_)
