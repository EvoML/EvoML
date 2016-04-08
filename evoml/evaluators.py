import random
from random import randint
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error
import math
from sklearn.cross_validation import train_test_split

def evalOneMax(individual, y_tr, x_te, y_te, base_estimator):
    predict_vals = []
    for i in range(0,len(individual)):
        chromosome = individual[i]
        predict_vals.append(get_predictions(chromosome,y_tr, x_te, base_estimator))
    final_prediction = []
    for i in range(0,len(individual)):
    	if(i==0):
    		final_prediction = predict_vals[0]
    	else:
    		final_prediction = final_prediction+predict_vals[i]
    final_prediction = final_prediction/len(individual)
    return math.sqrt(mean_squared_error(y_te, final_prediction)),

def get_predictions(chrom,y_tr, x_te, base_estimator):
    feat_chrom = list(chrom.columns.values)
    test_feat = x_te[feat_chrom]
    #mod = linear_model.LinearRegression()
    mod = base_estimator
    mod.fit(chrom, y_tr)
    predicted = mod.predict(test_feat)
    return predicted;