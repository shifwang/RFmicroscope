from abc import ABC, abstractmethod
from .sklearn_interface import all_tree_paths
import sklearn
import numpy as np
import pandas

class Reader(ABC):
    """docstring for Reader."""
    def __init__(self, **kwargs):
        super(Reader, self).__init__()


    @abstractmethod
    def read_from(self, obj):
        pass
    @abstractmethod
    def summary(self):
        pass
    @abstractmethod
    def reset(self):
        pass
class TreeReader(Reader):
    """docstring for TreeReader."""
    def __init__(self, **kwargs):
        super(Reader, self).__init__()
        self.info_ = None
        self.number_of_rows_ = 0


    @abstractmethod
    def read_from(self, obj):
        pass
    def summary(self):
        print('Here is the summary.')
        print('Number of features is %d'%(self.number_of_features_))
        print('Number of samples is %d'%(self.number_of_samples_))
        print('Number of paths is %d'%(self.number_of_rows_))
        print('Some samples from self.info_')
        print(self.info_.sample(min(10, self.number_of_rows_)))
        print('end.')

    def reset(self):
        if self.info_:
            self.info_.drop(self.info_.index, inplace = True)
        self.number_of_rows_ = 0



class OrdinaryTreeReader(TreeReader):
    """docstring for OrdinaryTreeReader."""
    def __init__(self,**kwargs):
        super(OrdinaryTreeReader, self).__init__(**kwargs)

        assert 'feature_names' in kwargs, \
        'OrdinaryTreeReader __init__ must specific feature_names!'
        assert 'sample_names' in kwargs, \
        'OrdinaryTreeReader __init__ must specific sample_names!'

        self.colnames_ = []

        self.feature_names_ = kwargs['feature_names']
        self.number_of_features_ = len(self.feature_names_)
        self.colnames_ += self.feature_names_

        self.sample_names_ = kwargs['sample_names']
        self.number_of_samples_ = len(self.sample_names_)
        self.colnames_ += self.sample_names_

        self.colnames_.append('tree_id')
        self.colnames_.append('leaf_id')

        self.colnames_.append('pred_label')
        self.info_ = pandas.DataFrame(columns = self.colnames_)

        self.number_of_rows_ = self.info_.shape[0]
    def read_from(self, tree, X, tree_id = 0):
        # tree must be of the type:
        assert type(tree) == sklearn.tree.tree.DecisionTreeClassifier,\
            'The type of tree must be sklearn.tree.tree.DecisionTreeClassifier but %s given.'%str(type(tree))

        # X must have the correct shape
        assert X.shape == (self.number_of_samples_, self.number_of_features_),\
            'The shape of X is not (%d, %d)'%(self.number_of_features_, self.number_of_samples_)

        # read all paths from tree
        paths = all_tree_paths(tree)

        # get prediction nodes
        pred_nodes = tree.tree_.apply(np.array(X, dtype = np.float32))

        # get prediction labels
        pred_labels = tree.predict(X)

        new_record = {colname:[] for colname in list(self.info_)}
        # add all paths into info
        for path in paths:

            new_record['tree_id'].append(tree_id)
            new_record['leaf_id'].append(path[-1])
            new_record['pred_label'].append(None)
            for f in self.feature_names_:
                new_record[f].append(False)
            for s in self.sample_names_:
                new_record[s].append(False)
            for node_ind in path[:-1]:
                new_record[self.feature_names_[tree.tree_.feature[node_ind]]][-1] = True
            for sample_ind in range(self.number_of_samples_):
                if pred_nodes[sample_ind] == path[-1]:
                    new_record[self.sample_names_[sample_ind]][-1] = True
                    new_record['pred_label'][-1] = pred_labels[sample_ind]
            #print(new_record)
        self.info_ = self.info_.append(pandas.DataFrame(new_record), ignore_index = True)
        self.number_of_rows_ = self.info_.shape[0]

class ImportanceTreeReader(TreeReader):
    """docstring for ImportanceTreeReader."""
    def __init__(self,**kwargs):
        super(ImportanceTreeReader, self).__init__(**kwargs)

        assert 'feature_names' in kwargs, \
        'ImportanceTreeReader __init__ must specific feature_names!'
        assert 'sample_names' in kwargs, \
        'ImportanceTreeReader __init__ must specific sample_names!'

        self.colnames_ = []

        self.feature_names_ = kwargs['feature_names']
        self.number_of_features_ = len(self.feature_names_)
        self.colnames_ += self.feature_names_

        self.sample_names_ = kwargs['sample_names']
        self.number_of_samples_ = len(self.sample_names_)
        self.colnames_ += self.sample_names_

        self.colnames_.append('tree_id')
        self.colnames_.append('leaf_id')

        self.colnames_.append('pred_label')
        self.info_ = pandas.DataFrame(columns = self.colnames_)

        self.number_of_rows_ = self.info_.shape[0]
    def read_from(self, tree, X, tree_id = 0):
        # tree must be of the type:
        assert type(tree) == sklearn.tree.tree.DecisionTreeClassifier,\
            'The type of tree must be sklearn.tree.tree.DecisionTreeClassifier but %s given.'%str(type(tree))

        # X must have the correct shape
        assert X.shape == (self.number_of_samples_, self.number_of_features_),\
            'The shape of X is not (%d, %d)'%(self.number_of_features_, self.number_of_samples_)

        # read all paths from tree
        paths = all_tree_paths(tree)

        # get prediction nodes
        pred_nodes = tree.tree_.apply(np.array(X, dtype = np.float32))

        # get prediction labels
        pred_labels = tree.predict(X)

        new_record = {colname:[] for colname in list(self.info_)}
        # add all paths into info
        for path in paths:

            new_record['tree_id'].append(tree_id)
            new_record['leaf_id'].append(path[-1])

            new_record['pred_label'].append(None)
            for f in self.feature_names_:
                new_record[f].append(0)
            for s in self.sample_names_:
                new_record[s].append(False)
            for sample_ind in range(self.number_of_samples_):
                if pred_nodes[sample_ind] == path[-1]:
                    new_record[self.sample_names_[sample_ind]][-1] = True
                    new_record['pred_label'][-1] = pred_labels[sample_ind]
            for i in range(len(path) - 1):
                node_ind = path[i]
                next_node_ind = path[i+1]
                #label = int(new_record['pred_label'][-1])
                label = 1
                prob_prev = tree.tree_.value[node_ind][0, label]/sum(tree.tree_.value[node_ind][0,:])
                prob_next = tree.tree_.value[next_node_ind][0, label]/sum(tree.tree_.value[next_node_ind][0,:])
                new_record[self.feature_names_[tree.tree_.feature[node_ind]]][-1] += prob_next - prob_prev

            #print(new_record)
        self.info_ = self.info_.append(pandas.DataFrame(new_record), ignore_index = True)
        self.number_of_rows_ = self.info_.shape[0]







class ForestReader(Reader):
    """ docstring for ForestReader"""
    def __init__(self, **kwargs):
        self.all_trees_ = None
        if 'number_of_features' in kwargs:
            self.number_of_features_ = kwargs['number_of_features']
        else:
            self.number_of_features_ = 0
        if 'number_of_samples' in kwargs:
            self.number_of_samples_ = kwargs['number_of_samples']
        else:
            self.number_of_samples_ = 0
        if 'feature_names' in kwargs:
            self.feature_names_ = kwargs['feature_names']
        elif self.number_of_features_ > 0:
            self.feature_names_ =  ['f%d'%i for i in range(self.number_of_features_)]
        else:
            self.feature_names_ = None
        if 'sample_names' in kwargs:
            self.sample_names_ =   kwargs['sample_names']
        elif self.number_of_samples_ > 0:
            self.sample_names_ =   ['s%d'%i for i in range(self.number_of_samples_)]
        else:
            self.sample_names_ = None
    def read_from(self, forest, X, TreeReaderType = 'Ordinary'):
        # sanity CHECK
        assert type(forest) == sklearn.ensemble.forest.RandomForestClassifier,\
            'The type of forest must be sklearn.ensemble.forest.RandomForestClassifier but %s given.'%str(type(forest))
        if self.number_of_samples_ == 0:
            self.number_of_samples_ = X.shape[0]
            self.sample_names_ =   ['s%d'%i for i in range(self.number_of_samples_)]
        elif self.number_of_samples_ != X.shape[0]:
            raise ValueError('X.shape[0] not equal to number_of_samples_')
        if self.number_of_features_ == 0:
            self.number_of_features_ = X.shape[1]
            self.feature_names_ =  ['f%d'%i for i in range(self.number_of_features_)]
        elif self.number_of_features_ != X.shape[1]:
            raise ValueError('X.shape[1] not equal to number_of_features_')
        # Main body
        self.TreeReaderType = TreeReaderType
        if TreeReaderType == 'Ordinary':
            treeReader = OrdinaryTreeReader(feature_names=self.feature_names_,
            sample_names = self.sample_names_)
        elif TreeReaderType == 'Importance':
            treeReader = ImportanceTreeReader(feature_names=self.feature_names_,
            sample_names = self.sample_names_)
        elif TreeReaderType == 'Signed':
            treeReader = SignedTreeReader()
        else:
            raise ValueError('TreeReaderType not valid (got %s)'%TreeReaderType)
        tree_id = 0
        for tree in forest.estimators_:
            treeReader.read_from(tree, X, tree_id)
            tree_id += 1
        self.all_trees_ = treeReader
    def summary(self):
        self.all_trees_.summary()
    def reset(self):
        self.all_trees_.reset()
        self.all_trees_ = None




if __name__ == '__main__':
    options = dict()
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    raw_data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        raw_data.data, raw_data.target, train_size=0.9,
        random_state=2017)
    rf = RandomForestClassifier(
        n_estimators=3, random_state=1231)
    rf.fit(X=X_train, y=y_train)
    e0 = rf.estimators_[0]
    a = OrdinaryTreeReader(feature_names = ['f%d'%i for i in range(30)],
    sample_names = ['s%d'%i for i in range(512)])
    a.read_from(e0, X_train)
    a.summary()
    b = ForestReader()
    b.read_from(rf, X_train,TreeReaderType = 'Importance')
    b.summary()
