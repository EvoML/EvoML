# -*- coding: utf-8 -*-
"""
Copyright 2016 Bhanu Pratap and Harsh Nisar.

This file is part of the Evoml library. 

The Evoml library is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License v3 or later.

Check the licesne file recieved along with the software for further details.
"""


import random
import numpy as np
import pandas as pd
from .util import EstimatorGene



def segment_mutator_EG(individual, pool_data, indpb, private_test = False):

    """
    Takes data from pool_data and mutuates existing training data
    to generate new fit estimators.

    Mutate can be:
     - add rows from pool_data randomly
     - delete rows randomly from the individual
     - replace a few rows from that of df 

    
    Parameters
    ----------
    individual: List of EstimatorGene

    pool_data : DataFrame
        Pool data from which rows are added or swapped.

    indpb : float, required
        Probablity with which mutuation happens on each EstimatorGene of the
    individual.
    """
    
    df_train = pool_data

    for i, eg_ in enumerate(individual):
        if random.random()>=indpb:
            continue
        # play around with tenpercent of current data.
        df_ = eg_.get_data()

        n_rows = int(0.05*pool_data.shape[0])

        rnd = random.random()
        if rnd<0.33:
            #add rows from the main df
            
            rows = np.random.choice(df_train.index.values, n_rows)
            df_ = df_.append(df_train.ix[rows])
        elif rnd<0.66:
            # delete rows randomly from the individual
            new_shape = df_.shape[0] - n_rows
            df_ = df_.sample(n=new_shape, replace = False, axis = 0)
            # df_.drop(labels=np.random.choice(df_.index, n_rows), axis=0, inplace=True)
        else:
            #replace a few rows
            new_shape = df_.shape[0] - n_rows
            df_ = df_.sample(n=new_shape, replace = False, axis = 0)
            # df_.drop(labels=np.random.choice(df_.index, n_rows), axis=0, inplace=True)
            rows = np.random.choice(df_train.index.values, n_rows)
            df_ = df_.append(df_train.ix[rows])
        
        ## Retrain the model in EstimatorGene with new data.
        eg_ =  EstimatorGene(df_.iloc[:,:-1], df_.iloc[:,-1], eg_.base_estimator, private_test = private_test)
        individual[i] = eg_

    
    return (individual,)


# def segment_mutator_EG_PT(individual, pool_data, indpb):
#     """
#     Takes data from pool_data and mutuates existing training data
#     to generate new fit estimators.
    
#     Parameters
#     ----------
#     individual: List of estimators.

#     Mutate can be:
#      - add rows from pool_data randomly
#      - delete rows randomly from the individual
#      - replace a few rows from that of df 
#     """
#     df_train = pool_data
    
#     for i, eg_ in enumerate(individual):
#         if random.random()>=indpb:
#             continue
#         # play around with tenpercent of current data.
#         df_ = eg_.get_data()

#         n_rows = int(0.05*pool_data.shape[0])
#         rnd = random.random()
#         if rnd<0.33:
#             #add rows from the main df
#             rows = np.random.choice(df_train.index.values, n_rows)
#             df_ = df_.append(df_train.ix[rows])
#         elif rnd<0.66:
#             # delete rows randomly from the individual
#             # issue with using drop is that all rows with the same index get deleted.
#             new_shape = df_.shape[0] - n_rows
#             df_ = df_.sample(n=new_shape, replace = False, axis = 0)
#             # This should be as good as deleting.
#             # df_.drop(labels=np.random.choice(df_.index, n_rows), axis=0, inplace=True)
#         else:
#             #replace a few rows
#             new_shape = df_.shape[0] - n_rows
#             df_ = df_.sample(n=new_shape, replace = False, axis = 0)
#             # df_.drop(labels=np.random.choice(df_.index, n_rows), axis=0, inplace=True)
#             rows = np.random.choice(df_train.index.values, n_rows)
#             df_ = df_.append(df_train.ix[rows])
        
#         ## Retrain the model in EstimatorGene with new data.
#         eg_ =  EstimatorGene(df_.iloc[:,:-1], df_.iloc[:,-1], eg_.base_estimator, private_test = True )
#         individual[i] = eg_

    
#     return (individual,)