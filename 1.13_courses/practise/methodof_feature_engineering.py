import pandas as pd
import numpy as np
from numpy import vstack, array, nan
from sklearn.datasets import load_iris

from sklearn import preprocessing
from sklearn import feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


iris =load_iris()

features = iris.data 
labels = iris.target 


feature_new = preprocessing.StandardScaler().fit_transform(features)

# print(feature_new)

feature_new = preprocessing.MinMaxScaler().fit_transform(features)

# print(feature_new)

feature_new = preprocessing.StandardScaler().fit_transform(features)
# print(feature_new)

feature_new=preprocessing.Binarizer(threshold=3.0).fit_transform(features)

# print(feature_new)
# rfe=feature_selection.RFE(estimator=LogisticRegression(),n_features_to_select=3).fit_transform(features,labels)
# feature_new = rfe.fit_transform(labels,features) 
# print(rfe.ranking_)


feature_new = feature_selection.SelectFromModel(GradientBoostingClassifier()).fit_transform(features,labels)

print(feature_new)



feature_new = 