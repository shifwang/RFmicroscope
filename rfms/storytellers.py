from rfms.readers import *
import numpy as np
import pandas as pd
from sklearn.ensemble.forest import _generate_unsampled_indices
def test_feature_importance(rf, X_test, y_test, loss = 'gini'):
    b = ForestReader()
    b.read_from(rf, X_test, TreeReaderType = 'Importance')
    out, feature_importances_, SE = individual_signed_feature_importance(b, y_test)
    return feature_importances_, SE
    
def oob_feature_importance(rf, X_train, y_train, loss = 'gini'):
    # generate importance forest reader
    reader = ForestReader()
    reader.read_from(rf, X_train, TreeReaderType = 'Importance')
    # generate feature importance template
    n_features = len(reader.feature_names_)
    n_samples = len(reader.sample_names_)
    
    # loop through each leaf, 
    #   for any sample it contains, add the feature importance
    path_sample = np.array(reader.all_trees_.info_[reader.sample_names_], dtype=float)
    for ind in range(len(rf.estimators_)):
        estimator = rf.estimators_[ind]
        unsampled_indices = _generate_unsampled_indices(estimator.random_state, n_samples)
        sampled_indices = np.array([True] * n_samples)
        sampled_indices[unsampled_indices] = 0
        sampled_indices = np.where(sampled_indices)[0]
        tmp = reader.all_trees_.info_.loc[reader.all_trees_.info_['tree_id'] == ind].index.values
        path_sample[np.ix_(tmp, sampled_indices)] = path_sample[np.ix_(tmp, sampled_indices)] * 0 # FIXME: the indexing
    useful_samples = np.sum(path_sample, 0) > 0
    path_sample[:,useful_samples] = path_sample[:, useful_samples] / np.sum(path_sample[:,useful_samples], 0)
    
    path_feature = np.array(reader.all_trees_.info_[reader.feature_names_], dtype=float)
    oob_individual_importance = path_sample.T.dot(path_feature)[useful_samples,:]
    tmp = (y_train[np.newaxis, useful_samples] - .5) * 2
    out = tmp @ oob_individual_importance / np.sum(useful_samples)
    out = out.flatten()
    if np.sum(out[out > 0]) + 10 * np.sum(out[out < 0]) < 0:
        return out
    else:
        return out / np.sum(out)

def overall_prevalence(forestReader, features, weighted_by_nodesize = True):
    ''' Compute the prevlance of a set of features
    '''
    if forestReader.TreeReaderType != 'Ordinary':
        raise AssertionError('TreeReaderType must be Ordinary but got %s'%forestReader.TreeReaderType)
    # initialize a dataframe
    if len(features) == 0:
        raise AssertionError('feature_names cannot be of length 0.')
    if type(features[0]) is not str:
        try:
            features = [forestReader.feature_names_[x] for x in features]
        except:
            raise ValueError('tried to convert features to strings but failed.')
    path_feature = np.array(forestReader.all_trees_.info_[features])
    if weighted_by_nodesize:
        path_sample = np.array(forestReader.all_trees_.info_[forestReader.sample_names_])
        path_weight = np.sum(path_sample,1) / np.sum(path_sample)
    else:
        path_weight = np.ones((path_feature.shape[0],)) / path_feature.shape[0]
    return np.sum(path_weight[np.mean(path_feature, 1) == 1])
    
def individual_signed_feature_importance(forestReader, labels = None):
    '''Compute the feature importance for each sample
    Args:
        forestReader - ImportanceTreeReader, the forestReader learned
        labels       - numpy array, the label of samples fed into forestReader, default None, only used to calculate the feature importance
    Returns:
        out - dataframe, sample * feature
        feature_importance - numpy 1d array, estimated feature importance
        importance_SE - numpy 1d array, standard error
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
    path_sample = np.array(forestReader.all_trees_.info_[forestReader.sample_names_])
    path_feature = np.array(forestReader.all_trees_.info_[forestReader.feature_names_])
#    for ind in forestReader.all_trees_.info_.index: # TODO: make sure it is each row
#        to_add = forestReader.all_trees_.info_.loc[ind, forestReader.feature_names_]
#        for sample in out.index:
#            if forestReader.all_trees_.info_.loc[ind,sample]:
#                out.loc[sample,:] += to_add
    number_of_trees_ = len(np.unique(forestReader.all_trees_.info_['tree_id']))
#    for sample in out.index:
#        out.loc[sample,:] = out.loc[sample,:] / number_of_trees_
    tmp_out = path_sample.T.dot(path_feature) / number_of_trees_
#    print(tmp_out - np.array(out), len(forestReader.sample_names_))
    out = pd.DataFrame(data=tmp_out, columns=forestReader.feature_names_, index = forestReader.sample_names_)
    if labels is not None:
        if set(labels[:]) != set([0, 1]):
            raise ValueError('only supported 0-1 label right now.')
        tmp = np.array(out, dtype=float) * (2 * labels - 1).reshape((len(labels), 1))
        feature_importance = np.mean(tmp, 0)
        importance_SE = np.std(tmp, 0) / (tmp.shape[0]) ** .5
        feature_importance, importance_SE = feature_importance / sum(feature_importance), importance_SE / sum(feature_importance)
    else:
        feature_importance, importance_SE = None, None
    return out, feature_importance, importance_SE
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
