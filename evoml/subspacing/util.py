import random
from random import randint
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error
import math
from sklearn.cross_validation import train_test_split
from sklearn.base import clone

def compare_hof(ind1, ind2):
    return np.all(ind1.fitness==ind2.fitness)

class EstimatorGene:
    """ Last column has to be the dependent variable"""
    def __init__(self, X, y, X_test, y_test, base_estimator):
       self.base_estimator = base_estimator
       self.estimator = clone(base_estimator)
       self.estimator.fit(X, y)
       self.X = X
       self.y = y

    def get_data(self):
       return pd.concat([self.X, self.y], axis = 1)