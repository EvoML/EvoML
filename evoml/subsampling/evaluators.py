# -*- coding: utf-8 -*-
"""
Copyright 2016 Bhanu Pratap and Harsh Nisar.

This file is part of the Evoml library. 

The Evoml library is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License v3 or later.

Check the licesne file recieved along with the software for further details.
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from util import centroid_df
from util import distance

def evalOneMax_KNN(individual, df_test, base_estimator):
    """
    This will have the code for testing the model on unseen data.
    
    Parameters
    ----------
    individual : list, required
        List of dataframes with each dataframes last column the
        dependent column.
    
    df_test : DataFrame, required
        The test dataframe against which you are evaluating your
        model in the fitness function.

    model : Estimator, required
        The estimator to use in the model.


    Side note: We can even start with just one partial part of the dataset, keep trying to
    increase its cv score in fitness function. To get the optimal dataset. I am fearing, 
    it will make a dataset of exactly same values in the end. That will have the best cv error.
    Therefore need to add a component of unseen data performance.
    """
    total_mse = []
    ensembles = []
    centroids = []

    for df_ in individual:
        # clf = LinearRegression()
        clf = base_estimator()

        clf = clf.fit(df_.iloc[:, 0:-1], df_.iloc[:,-1])
        ensembles.append(clf)
        centroids.append(centroid_df(df_.iloc[:,0:-1]))
        ## for sum of mse return uncomment these
        #y_pred = clf.predict(df_test[x_columns])
        #mse = mean_squared_error(y_pred, df_test[y_column])
        #total_mse.append(mse)
    
    #print total_mse
    y_preds_ensemble = []
    for row in df_test.iloc[:, 0:-1].values:
        distances = [distance(row, centroid) for centroid in centroids]
        model_index = np.argmin(distances)
        #todo: optional use the average of the 2 min distances ka prediction.
        #todo: parameter.
        y_pred = ensembles[model_index].predict(row)[0]
        

        y_preds_ensemble.append(y_pred)

    rmse_ensemble = np.sqrt(mean_squared_error(y_preds_ensemble, df_test.iloc[:,-1]))
    return (rmse_ensemble),

def evalOneMax_KNN_EG(individual, df_test, base_estimator, n_votes):
    """
    This will have the code for testing the model on unseen data.
    
    Parameters
    ----------
    individual : list, required
        List of dataframes with each dataframes last column the
        dependent column.
    
    df_test : DataFrame, required
        The test dataframe against which you are evaluating your
        model in the fitness function.

    model : Estimator, required
        The estimator to use in the model.


    Side note: We can even start with just one partial part of the dataset, keep trying to
    increase its cv score in fitness function. To get the optimal dataset. I am fearing, 
    it will make a dataset of exactly same values in the end. That will have the best cv error.
    Therefore need to add a component of unseen data performance.
    """
    total_mse = []
    ensembles = []
    centroids = []

    for eg_ in individual:
        # clf = LinearRegression()
        
        clf = eg_.estimator

        ensembles.append(clf)
        
        centroids.append(centroid_df(eg_.X))
        
        ## for sum of mse return uncomment these
        #y_pred = clf.predict(df_test[x_columns])
        #mse = mean_squared_error(y_pred, df_test[y_column])
        #total_mse.append(mse)
    
    #print total_mse
    y_preds_ensemble = []
    ensembles = np.array(ensembles)

    # df_test = df_test.sample(, replace = False)

    for row in df_test.iloc[:, 0:-1].values:
        distances = np.array([distance(row, centroid) for centroid in centroids])
        # model_index = np.argmin(distances)
        #todo: optional use the average of the 2 min distances ka prediction
        #todo: parameter.
        
        
        model_ixs = distances.argsort()[:n_votes]
        

        models = ensembles[model_ixs]


        # mean of all predictions.
        y_pred = np.nanmean([mdl.predict(row)[0] for mdl in models])
    
        # y_pred = ensembles[model_index].predict(row)[0]
        
        y_preds_ensemble.append(y_pred)

    # rmse_ensemble = np.sqrt(mean_squared_error(y_preds_ensemble, df_test.iloc[:,-1]))
    rmse_ensemble = np.sqrt(mean_squared_error(y_preds_ensemble, df_test.iloc[:,-1]))
    return (rmse_ensemble),


def eval_ensemble_oob_KNN_EG(individual, df, base_estimator, n_votes):
    """
    Fitness is based on entire ensembles performance on Out of Bag samples
    calculated using the union of the all samples used while training each child model.

    Used in FEGO.

    Parameters
    ----------
    individual : list, required
        List of dataframes with each dataframes last column the
        dependent column.
    
    df: DataFrame, required
        The entire dataframe given to model to train - X and y combined.

    model : Estimator, required
        The estimator to use in the model.
        #todo: not really needed. double check.

    Side note: We can even start with just one partial part of the dataset, keep trying to
    increase its cv score in fitness function. To get the optimal dataset. I am fearing, 
    it will make a dataset of exactly same values in the end. That will have the best cv error.
    Therefore need to add a component of unseen data performance.
    """
    total_mse = []
    ensembles = []
    centroids = []

    # All the indices used in each child model.
    bag_idx = []

    for eg_ in individual:
        # clf = LinearRegression()
        
        clf = eg_.estimator

        ensembles.append(clf)
        
        idx = eg_.X.index.tolist()
        # print len(idx)
        bag_idx.append(idx)

        centroids.append(centroid_df(eg_.X))
        
        ## for sum of mse return uncomment these
        #y_pred = clf.predict(df_test[x_columns])
        #mse = mean_squared_error(y_pred, df_test[y_column])
        #total_mse.append(mse)
    
    #print total_mse
    # flattening and converting to set.
    
    # print bag_idx
    bag_idx = set(sum(bag_idx, []))
    # print len(bag_idx)
    out_bag_idx  = list(set(df.index.tolist()) - bag_idx)
    print 'Size of OOB', len(out_bag_idx)
    df_test = df.loc[out_bag_idx]

    y_preds_ensemble = []
    ensembles = np.array(ensembles)


    for row in df_test.iloc[:, 0:-1].values:
        distances = np.array([distance(row, centroid) for centroid in centroids])
        # model_index = np.argmin(distances)
        #todo: optional use the average of the 2 min distances ka prediction.
        #todo: parameter.
        
        
        model_ixs = distances.argsort()[:n_votes]
        

        models = ensembles[model_ixs]


        # mean of all predictions.
        y_pred = np.mean([mdl.predict(row)[0] for mdl in models])
            
        # y_pred = ensembles[model_index].predict(row)[0]
        
        y_preds_ensemble.append(y_pred)

    rmse_ensemble = np.sqrt(mean_squared_error(y_preds_ensemble, df_test.iloc[:,-1]))
    return (rmse_ensemble),


def eval_each_model_oob_KNN_EG(individual, df, base_estimator, n_votes):
    """
    Fitness is based on average of each constituent model's RMSE over its own
    private oob. 

    Used in FEMPO.

    Replicates how a random forest works.

    Does not consider voting in the fitness. (How will it figure out to make better distinguished
    segments?)

    Parameters
    ----------
    individual : list, required
        List of dataframes with each dataframes last column the
        dependent column.
    
    df: DataFrame, required
        The entire dataframe given to model to train - X and y combined.

    model : Estimator, required
        The estimator to use in the model.
        #todo: not really needed. double check.

    Side note: We can even start with just one partial part of the dataset, keep trying to
    increase its cv score in fitness function. To get the optimal dataset. I am fearing, 
    it will make a dataset of exactly same values in the end. That will have the best cv error.
    Therefore need to add a component of unseen data performance.
    """
    total_rmses = []
    ensembles = []
    centroids = []

    # All the indices used in each child model.
    bag_idx = []

    for eg_ in individual:
        
        # Generate Private OOB.
        bag_idx = set(eg_.X.index.tolist())
        out_bag_idx  = list(set(df.index.tolist()) - bag_idx)
        df_test = df.loc[out_bag_idx]        

        clf = eg_.estimator

        ensembles.append(clf)
        
        bag_idx = set(list(eg_.X.index.tolist()))

        p_out_bag_idx  = list(set(df.index.tolist()) - bag_idx)
        p_out_bag = df.loc[p_out_bag_idx]

        # print len(p_out_bag.columns)
        p_out_bag_X = p_out_bag.iloc[:,:-1]
        p_out_bag_y = p_out_bag.iloc[:,-1]
        # will test on p_out_bag
        if p_out_bag_y.shape[0] == 0:
            print 'OOB ran-out'
            #Should we then use rmse on itself?
            continue

        preds = clf.predict(p_out_bag_X)
        rmse = np.sqrt(mean_squared_error(p_out_bag_y, preds))
        # rmse = mean_squared_error(p_out_bag_y, preds)
        total_rmses.append(rmse)
        
        ## for sum of mse return uncomment these
        #y_pred = clf.predict(df_test[x_columns])
        #mse = mean_squared_error(y_pred, df_test[y_column])
        #total_mse.append(mse)
    
    #print total_mse
    # flattening and converting to set.
    return (np.nanmean(total_rmses)),

def eval_each_model_PT_KNN_EG(individual, df, base_estimator, n_votes):
    """
    Fitness is based on average of each constituent model's RMSE over its own
    private test_set

    Used in FEMPT.

    Replicates how a random forest works.

    Does not consider voting in the fitness. (How will it figure out to make better distinguished
    segments?)
    - Makes each segment robust.
    
    Parameters
    ----------
    individual : list, required
        List of dataframes with each dataframes last column the
        dependent column.
    
    df: DataFrame, required
        The entire dataframe given to model to train - X and y combined.

    model : Estimator, required
        The estimator to use in the model.
        #todo: not really needed. double check.

    Side note: We can even start with just one partial part of the dataset, keep trying to
    increase its cv score in fitness function. To get the optimal dataset. I am fearing, 
    it will make a dataset of exactly same values in the end. That will have the best cv error.
    Therefore need to add a component of unseen data performance.
    """
    total_rmses = []
    ensembles = []
    centroids = []

    # All the indices used in each child model.
    bag_idx = []

    for eg_ in individual:
        
        # Generate Private OOB.

        clf = eg_.estimator

        ensembles.append(clf)
        
        # rmse is precalculated in EG if private_test = True
        rmse = eg_.rmse
        total_rmses.append(rmse)
        
        ## for sum of mse return uncomment these
        #y_pred = clf.predict(df_test[x_columns])
        #mse = mean_squared_error(y_pred, df_test[y_column])
        #total_mse.append(mse)
    
    #print total_mse
    # flattening and converting to set.
    return (np.nanmean(total_rmses)),


    