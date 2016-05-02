import numpy as np
import pandas as pd
import random
# from mutators import segment_mutator

from evaluators import evalOneMax_KNN

from util import centroid_df
from util import distance

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split



def segment_mutator(individual, pool_data, indpb):
    """
    Takes data from pool_data and mutuates each dataframe
    in the individual.

    Mutate can be:
     - add rows from pool_data randomly
     - delete rows randomly from the individual
     - replace a few rows from that of df 
    """
    df_train = pool_data
    
    for i, df_ in enumerate(individual):
        if random.random()>=indpb:
            continue
        # play around with tenpercent of current data.
        
        n_rows = int(0.2*df_.shape[0])
        rnd = random.random()
        if rnd<0.33:
            #add rows from the main df
            
            rows = np.random.choice(df_train.index.values, n_rows)
            df_ = df_.append(df_train.ix[rows])
        elif rnd<0.66:
            # delete rows randomly from the individual
            df_.drop(labels=np.random.choice(df_.index, n_rows), axis=0, inplace=True)
        else:
            #replace a few rows
            df_.drop(labels=np.random.choice(df_.index, 1), axis=0, inplace=True)
            rows = np.random.choice(df_train.index.values, n_rows)
            df_ = df_.append(df_train.ix[rows])
        
        individual[i] = df_.copy()

    
    return (individual,)


def get_df_sample(sample_percentage, pool_data):
    return pool_data.sample(frac=sample_percentage)

def similar_individual(ind1, ind2):
    return np.all(ind1.fitness.values == ind2.fitness.values)

class BasicSegmenter(BaseEstimator, RegressorMixin):
    """
    Uses basic evolutionary algorithm to find the best subsets of X and trains
    Linear Regression on each subset. For given row of input, prediction
    is based on the model trained on segment closest to input.

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

    def __init__(self, n = 10, base_estimator = LinearRegression, test_size = 0.2, 
                n_population = 30, cxpb=0.5, mutpb=0.5, ngen=50, tournsize = 3, 
                init_sample_percentage = 0.2, indpb =0.20, crossover_func = tools.cxTwoPoint):
        
        self.n = n
        self.test_size = test_size
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen
        self.tournsize = tournsize
        self.init_sample_percentage = init_sample_percentage
        self.base_estimator = base_estimator
        self.indpb = indpb
        self.n_population = n_population
        self.crossover_func = crossover_func

    def fit(self, X, y):
        # Is returning NDFrame and hence errors in concat.
        # X, y = check_X_y(X, y)
        self.X_ = X
    
        self._X_mean = X.mean()
        self._X_std = X.std()

        X = (X - self._X_mean)/self._X_std


        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=self.test_size, 
                                                            random_state = 64)

        
        df_train = pd.concat([pd.DataFrame(X_train), y_train], axis = 1)
        df_test = pd.concat([X_test, y_test], axis = 1)

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

        toolbox.register("df_sample", get_df_sample, self.init_sample_percentage, df_train)

        ## Thinking what our individual will be? A list of scikit mdls, a list of dataframes, or a mixed class.
        ## Evaluations on individuals are saved and not done again if the fitness remains unchanged.
        ## In that case models don't need to created again, but they need to be saved for evaluati

        # n = 10, defines an ensemble of ten. #todo: Can push the parameter uptop later 
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.df_sample, self.n)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)


        toolbox.register("evaluate", evalOneMax_KNN, df_test = df_test, base_estimator = self.base_estimator)
        toolbox.register("mate", self.crossover_func)
        toolbox.register("mutate", segment_mutator, pool_data = df_train, indpb = self.indpb)
        toolbox.register("select", tools.selTournament, tournsize= self.tournsize)


        pop = toolbox.population(n = self.n_population)
       
        hof = tools.HallOfFame(1, similar=similar_individual)
        #hof = tools.ParetoFront(similar=similar_individual)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        #stats = tools.Statistics(lambda ind: [x.shape[0] for x in ind])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        algorithms.eaSimple(pop, toolbox, cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.ngen, stats=stats, halloffame= hof)

        self.segments_ = hof[0]
        
        # print self.segments_

        #should  be setting these pop, stats, hof

        return self

    def predict(self, X):

        ensembles = []
        centroids = []

        # scaling using the mean and std of the original train data.
        X = (X - self._X_mean)/self._X_std

        for df_ in self.segments_:
            # clf = LinearRegression()
            clf = self.base_estimator()

            clf = clf.fit(df_.iloc[:,0:-1], df_.iloc[:,-1])
            ensembles.append(clf)
            centroids.append(centroid_df(df_.iloc[:,0:-1]))
            ## for sum of mse return uncomment these
            #y_pred = clf.predict(df_test[x_columns])
            #mse = mean_squared_error(y_pred, df_test[y_column])
            #total_mse.append(mse)
        
        #print total_mse
        y_preds_ensemble = []
        for row in X.values:
            distances = [distance(row, centroid) for centroid in centroids]
            model_index = np.argmin(distances)
            #todo: optional use the average of the 2 min distances ka prediction.
            y_pred = ensembles[model_index].predict(row)[0]
            

            y_preds_ensemble.append(y_pred)
        
        return pd.Series(y_preds_ensemble)    





