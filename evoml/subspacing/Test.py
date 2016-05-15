import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.base import clone
import sklearn.datasets as sk_data
import matplotlib.pyplot as plt
boston = load_boston()
from operator import attrgetter
from sklearn.tree import DecisionTreeRegressor
from .feature_stacker_FEGT import FeatureStackerFEGT
from .feature_stacker_FEMPO import FeatureStackerFEMPO

b_feat = pd.DataFrame(boston.data)
b_feat.columns = ['feat_0','feat_1','feat_2','feat_3','feat_4','feat_5','feat_6','feat_7','feat_8','feat_9','feat_10','feat_11','feat_12']
b_target = pd.DataFrame(boston.target)
b_target.columns = ['output']

# FEGT
FS1 = FeatureStackerFEGT()
FS1.fit(b_feat, b_target)

# FEMPO
FS2 = FeatureStackerFEMPO()
FS2.fit(b_feat, b_target)