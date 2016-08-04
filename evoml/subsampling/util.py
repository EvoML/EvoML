# -*- coding: utf-8 -*-
"""
Copyright 2016 Bhanu Pratap and Harsh Nisar.

This file is part of the Evoml library. 

The Evoml library is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License v3 or later.

Check the licesne file recieved along with the software for further details.
"""

import numpy as np
from sklearn.base import clone
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

def centroid_df(df):
    """ returns the mean of all the points in the df as a vector.
    this will be the centroid. The dataset should be scaled before hand. 
    """
    return df.mean().values

def distance(row1, row2):
    """Distance between two vectors """
    dist = np.linalg.norm(row1 - row2)
    return dist

class EstimatorGene:
    """ Last column has to be the dependent variable"""
    def __init__(self, X, y, base_estimator, private_test = False):
        self.base_estimator = base_estimator
        self.estimator = clone(base_estimator)
        self.X = X
        self.y = y
        
        if private_test == True:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 12)
            self.estimator.fit(X_train, y_train)
            self.rmse = np.sqrt(mean_squared_error(self.estimator.predict(X_test), y_test))
        else:
            self.estimator.fit(X, y)
            


    def get_data(self):
        return pd.concat([self.X, self.y], axis = 1)
