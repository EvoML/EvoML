
"""
Algorithm: FEMPO
Fitness each model private oob.

The fitness function of ensemble is the average of the RMSE of each child model over private
oob set for respective models. 

mutators: same as before.
"""
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
import random
# from mutators import segment_mutator

from evaluators import eval_each_model_oob_KNN_EG
from util import EstimatorGene

from util import centroid_df
from util import distance

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

        

def segment_mutator_EG(individual, pool_data, indpb):
    """
    Takes data from pool_data and mutuates existing training data
    to generate new fit estimators.
    
    Parameters
    ----------
    individual: List of estimators.

    Mutate can be:
     - add rows from pool_data randomly
     - delete rows randomly from the individual
     - replace a few rows from that of df 
    """
    df_train = pool_data
    
    for i, eg_ in enumerate(individual):
        if random.random()>=indpb:
            continue
        df_ = eg_.get_data()

        # play around with tenpercent of current data.
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
        eg_ =  EstimatorGene(df_.iloc[:,:-1], df_.iloc[:,-1], eg_.base_estimator)
        individual[i] = eg_

    
    return (individual,)


def get_mdl_sample(sample_percentage, pool_data, base_estimator):
    """ Returns an instance of EstimatorGene 
    fit with a random sample of the pool data.
    Assumes the last column is the dependent column.

    If sample percentage is given as None, then creates samples based 
    on random sample percentages.  
    """
    


    if sample_percentage == None:
        #TODO: Can parameterize this min and max range too. But for now, let it flow.
        sample_percentage = random.randrange(0.1, 1, _int=float)
    data = pool_data.sample(frac=sample_percentage, replace = True)
    X = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]
    return EstimatorGene(X, y, base_estimator)


def similar_individual(ind1, ind2):
    return np.all(ind1.fitness.values == ind2.fitness.values)


class BasicSegmenterEG_FEMPO(BaseEstimator, RegressorMixin):
    """
    Uses basic evolutionary algorithm to find the best subsets of X and trains
    Linear Regression on each subset. For given row of input, prediction
    is based on the model trained on segment closest to input.

    Same as the BasicSegmenter, but uses list of trained models instead of DataFrames
    as each individual. Done to boost performance. 

    Algorithm: FEMPO
    Fitness each model private oob.

    The fitness function of ensemble is the average of the RMSE of each child model over private
    oob set for respective models. 


    Parameters
    ----------
    n : Integer, optional, default, 10
        The number of segments you want in your dataset.
    
    base_estimator: estimator, default, LinearRegression
        The basic estimator for all segments.

    test_size : float, optional, default, 0.2
        Test size that the algorithm internally uses in its 
        fitness function.

    n_population : Integer, optional, default, 30
        The number of ensembles present in population.

    init_sample_percentage : float, optional, default, 0.2
    

    Attributes
    -----------
    best_enstimator_ : estimator 
    
    segments_ : list of DataFrames

    """

    def __init__(self, n = 10, test_size = 0.2, 
                n_population = 30, cxpb=0.5, mutpb=0.5, ngen=50, tournsize = 3, 
                init_sample_percentage = 0.2, indpb =0.20, crossover_func = tools.cxTwoPoint, statistics = True,
                base_estimator = LinearRegression(), n_votes = 1, unseen_x = None, unseen_y = None):
        
        self.n = n
        self.test_size = test_size
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen
        self.tournsize = tournsize
        self.init_sample_percentage = init_sample_percentage
        #self.base_estimator = base_estimator
        self.indpb = indpb
        self.n_population = n_population
        self.crossover_func = crossover_func
        self.statistics = statistics
        self.base_estimator = base_estimator
        self.n_votes = n_votes

        self.unseen_x = unseen_x
        self.unseen_y = unseen_y

    def fit(self, X, y):
        # Is returning NDFrame and hence errors in concat.
        # X, y = check_X_y(X, y)

        self.X_ = X
        self.y_ = y
        self._X_mean = X.mean()
        self._X_std = X.std()

        X = (X - self._X_mean)/self._X_std



#        X_train, X_test, y_train, y_test = train_test_split(X, y, 
#                                                           test_size=self.test_size)
        # There is no test created in this. Private oob is used.
        
        df = pd.concat([X, y], axis = 1)

        # print df_train.shape
        # print df_test.shape
        # #print df_train.columns
        
        # mdl = LinearRegression().fit(df_train[x_columns], df_train[y_column])
        # print df_train[y_column].ndim
        # mdl = LassoCV().fit(df_train[x_columns], df_train[y_column])
        # print np.sqrt(mean_squared_error(mdl.predict(df_test[x_columns]), df_test[y_column]))

        ### Setting toolbox
        creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list , fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        ## In this case we will also need to pass the base estimator.
        toolbox.register("mdl_sample", get_mdl_sample, self.init_sample_percentage, df, self.base_estimator)

        ## Thinking what our individual will be? A list of scikit mdls, a list of dataframes, or a mixed class.
        ## Evaluations on individuals are saved and not done again if the fitness remains unchanged.
        ## In that case models don't need to created again, but they need to be saved for evaluati

        # n = 10, defines an ensemble of ten. #todo: Can push the parameter uptop later 
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.mdl_sample, self.n)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)


        toolbox.register("evaluate", eval_each_model_oob_KNN_EG, df = df, base_estimator = self.base_estimator, n_votes = self.n_votes)
        toolbox.register("mate", self.crossover_func)
        toolbox.register("mutate", segment_mutator_EG, pool_data = df, indpb = self.indpb)
        toolbox.register("select", tools.selTournament, tournsize= self.tournsize)


        pop = toolbox.population(n = self.n_population)
       
        hof = tools.HallOfFame(1, similar=similar_individual)
        #hof = tools.ParetoFront(similar=similar_individual)

        def eval_unseen_per_gen(ind, unseen_x = self.unseen_x, unseen_y = self.unseen_y):
            """
            Unseen is taken from init params and is a complete  
            """
            ensembles = []
            centroids = []

            X = unseen_x.copy()
            # scaling using the mean and std of the original train data.
            X = (X - self._X_mean)/self._X_std
            for eg_ in ind:
                df_ = eg_.get_data()
                ensembles.append(eg_.estimator)
                centroids.append(centroid_df(df_.iloc[:,0:-1]))
            
            y_preds_ensemble = []
            ensembles = np.array(ensembles)
            for row in X.values:
                distances = np.array([distance(row, centroid) for centroid in centroids])
                model_ixs = distances.argsort()[:self.n_votes]
                models = ensembles[model_ixs]
                y_pred = np.mean([mdl.predict(row)[0] for mdl in models])
                y_preds_ensemble.append(y_pred)
                ## MSE
            
            return mean_squared_error(y_preds_ensemble, unseen_y)

        if self.statistics != None:
            # stats = tools.Statistics(lambda ind: ind.fitness.values)
            
            stats_fitness = tools.Statistics(key = lambda ind: ind.fitness.values)
            stats_unseen_performance = tools.Statistics(key = eval_unseen_per_gen)
            mstats = tools.MultiStatistics(fitness=stats_fitness, unseen = stats_unseen_performance)
            stats = tools.Statistics(eval_unseen_per_gen)
            
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            mstats.register("avg", np.mean)
            mstats.register("std", np.std)
            mstats.register("min", np.min)
            mstats.register("max", np.max)

        else:
            #None
            stats = self.statistics

        #stats = tools.Statistics(lambda ind: [x.shape[0] for x in ind])
        
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.ngen, stats=None, halloffame= hof, verbose = True)
        
        self.pop = pop
        self.log = log

        self.segments_ = hof[0]
        
        # print self.segments_

        #should  be setting these pop, stats, hof

        return self

    def predict(self, X):

        ensembles = []
        centroids = []

        # scaling using the mean and std of the original train data.
        X = (X - self._X_mean)/self._X_std

        for eg_ in self.segments_:
            
            df_ = eg_.get_data()
            # clf = LinearRegression()
            # clf = self.base_estimator()

            # clf = clf.fit(df_.iloc[:,0:-1], df_.iloc[:,-1])
            
            # EG.estimator is already fit with the internal data.
            ensembles.append(eg_.estimator)
            centroids.append(centroid_df(df_.iloc[:,0:-1]))
            ## for sum of mse return uncomment these
            #y_pred = clf.predict(df_test[x_columns])
            #mse = mean_squared_error(y_pred, df_test[y_column])
            #total_mse.append(mse)
        
        #print total_mse
        y_preds_ensemble = []
        ensembles = np.array(ensembles)
        for row in X.values:
            distances = np.array([distance(row, centroid) for centroid in centroids])
            # model_index = np.argmin(distances)
            #todo: optional use the average of the 2 min distances ka prediction.
            
            # get n_votes centroids closest to the row. 
            model_ixs = distances.argsort()[:self.n_votes]
            
            models = ensembles[model_ixs]

            # mean of all predictions.
            y_pred = np.mean([mdl.predict(row)[0] for mdl in models])
            

            y_preds_ensemble.append(y_pred)
        
        return pd.Series(y_preds_ensemble)    
