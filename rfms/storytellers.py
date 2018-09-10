from . import readers
import numpy as np
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
    if forestReader.TreeReaderType != 'ImportanceTreeReader':
        raise AssertionError('TreeReaderType must be ImportanceTreeReader')
    # initialize a dataframe
    out = {'sample_names': forestReader.sample_names_}
    for f in forestReader.feature_names_:
        out[f] = np.zeros((forestReader.number_of_samples_, ))
    out = pd.dataframe(out, index = 'sample_names')
    # loop through each leaf, 
    #   for any sample it contains, add the feature importance
    for ind in forestReader.info_.index: # TODO: make sure it is each row
        for sample in out.index:
            if forestReader.info_.loc[ind,sample]:
                out.loc[sample,:] += forestReader.info_.loc[ind, forestReader.feature_names_]
    for sample in out.index:
        out.loc[sample,:] = out.loc[sample,:] / number_of_trees
    return out
