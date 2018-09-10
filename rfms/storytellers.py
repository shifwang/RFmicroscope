from . import readers
import numpy as np
import pandas as pd
def individual_signed_feature_importance(forestReader):
    '''Compute the feature importance for each sample
    Args:
        forestReader - ImportanceTreeReader, the 
    Returns:
        out - dataframe, sample * feature
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
        for sample in out.index:
            if forestReader.all_trees_.info_.loc[ind,sample]:
                out.loc[sample,:] += forestReader.all_trees_.info_.loc[ind, forestReader.feature_names_]
    number_of_trees_ = len(np.unique(forestReader.all_trees_.info_['tree_id']))
    for sample in out.index:
        out.loc[sample,:] = out.loc[sample,:] / number_of_trees_
    return out
