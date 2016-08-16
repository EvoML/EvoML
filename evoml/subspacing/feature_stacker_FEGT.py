"""
Algorithm - Fitness_Ensemble_Global_Test (FEGT)
Fitness is measured that of the entire ensemble and on a global test set.
Global Test is calcuated in the starting where the test set for all the ensembles
is seggregated from the train set. The test set remains the same for each chromosome 
inside an individual and also for all the individuals.

Evolutionary Algorithm: Genetic Algorithm
Evaluator: evalOneMax
Mutator: mutate_feat
"""

import random
from random import randint
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error
import math
from sklearn.cross_validation import train_test_split
from .evaluators import evalOneMax
from .evaluators import evalOneMax2
from .evaluators import evalOneMax_class1
from .mutators import mutate_feat
from .util import compare_hof
from sklearn.base import BaseEstimator, RegressorMixin
from .util import EstimatorGene
from collections import Counter

class FeatureStackerFEGT(BaseEstimator,RegressorMixin):
    """
    Uses basic evolutionary algorithm to find the best subspaces of X and trains 
    a model on each subspace. For given row of input, prediction is based on the ensemble
    which has performed the best on the test set. The prediction is the average of all the 
    chromosome predictions.

    Same as the BasicSegmenter, but uses list of thrained models instead of DataFrames
    as each individual. Done to boost performance. 

    Parameters
    ----------
    test_size: float, default = 0.2
        Test size that the algorithm internally uses in its fitness
        function
    
    N_population: Integer, default : 30
        The population of the individuals that the evolutionary algorithm is going to use. 
    
    N_individual: Integer, default : 5
        Number of chromosomes in each individual of the population

    featMin: Integer, default : 1
        The minimum number of features for the sub space from the dataset
        Cannot be <= 0 else changes it to 1 instead.
    
    featMax: Integer, default : max number of features in the dataset
        The maximum number of features for the sub space from the dataset
        Cannot be <featMin else changes it to equal to featMin

    indpb: float, default : 0.05
        The number that defines the probability by which the chromosome will be mutated.

    ngen: Integer, default : 10
        The iterations for which the evolutionary algorithm is going to run.

    mutpb: float, default : 0.40
        The probability by which the individuals will go through mutation.

    cxpb: float, default : 0.50
        The probability by which the individuals will go through cross over.

    base_estimator: model, default: LinearRegression
        The type of model which is to be trained in the chromosome.

    crossover_func: cross-over function, default : tools.cxTwoPoint [go through eaSimple's documentation]
        The corssover function that will be used between the individuals

    test_frac, test_frac_flag: Parameters for playing around with test set. Not in use as of now.

    Attributes
    -----------
    segment: HallOfFame individual 
        Gives you the best individual from the whole population. 
        The best individual can be accessed via segment[0]

    """

    def __init__(self, test_size=0.20,N_population=30,N_individual=5,featMin=1,featMax=None,
        indpb=0.05, ngen = 10, mutpb = 0.40, cxpb = 0.50,
        base_estimator=linear_model.LinearRegression(), crossover_func = tools.cxTwoPoint,
        test_frac=0.30, test_frac_flag = False, model_type = 'regression', maxOrMin = 1, verbose_flag = True):
        # model_type defines whether its a regression method or a classification method
        self.test_size = test_size
        self.N_population = N_population
        self.N_individual = N_individual
        self.indpb = indpb
        self.ngen = ngen
        self.mutpb = mutpb
        self.cxpb = cxpb
        self.base_estimator = base_estimator
        self.crossover_func = crossover_func
        self.test_frac = test_frac
        self.test_frac_flag = test_frac_flag
        if(featMin<=0):
            featMin = 1
            print("featMin cannot be <=0, hence has been modified to 1")
        self.featMin = featMin
        if featMax is not None:
            if(featMax<featMin):
                featMax = featMin
                print("featMax cannot be <featMin, hence has been made equal to featMin")
        self.featMax = featMax        
        self.model_type = model_type
        self.maxOrMin = maxOrMin
        self.verbose_flag = verbose_flag

    def get_indiv_sample(self,data,output,base_estimator):
        '''
        The function generates the chromosomes for the population
        '''
        if self.featMax is None:
            self.featMax = data.shape[1]-1
        feat_name = list(data.columns.values)
        feat_count = (randint(self.featMin,self.featMax))
        ind = random.sample(range(0, self.featMax), feat_count)
        new_feat = []
        for i in range(0,len(ind)):
            new_feat.append(feat_name[ind[i]])
        # The below line is just to create a bagged chromosome for the 
        tmp_data = data[new_feat].sample(frac=1, replace=True)
        output_new = output.loc[list(tmp_data.index)]
        return EstimatorGene(tmp_data,output_new,[],[],base_estimator)

    def fit(self, X, y):
        '''
        The function takes as input:

        X: Features of the whole dataset
            All features which are to be used for building the model.

        y: The output which is to be predicted with the model. 
            The model is trained to predict these models via X.
        '''
        input_feat = list(X.columns.values);
        creator.create("FitnessMax", base.Fitness, weights=(self.maxOrMin,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        # The train and test sample for the
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", self.get_indiv_sample, data=X_train, output=y_train, base_estimator=self.base_estimator)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=self.N_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        if (self.model_type=='regression'):
            toolbox.register("evaluate", evalOneMax,x_te = X_test, y_te = y_test, test_frac = self.test_frac, test_frac_flag = self.test_frac_flag)
        elif (self.model_type=='classification'):
            toolbox.register("evaluate", evalOneMax_class1,x_te = X_test, y_te = y_test, test_frac = self.test_frac, test_frac_flag = self.test_frac_flag)
        toolbox.register("mate", self.crossover_func)
        toolbox.register("mutate", mutate_feat, indpb=self.indpb,input_fe = input_feat, X_tr = X_train)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        pop = toolbox.population(n=self.N_population)
        hof = tools.HallOfFame(1, similar=compare_hof);
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        self.pop, self.logbook = algorithms.eaSimple(pop, toolbox, cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.ngen, stats=stats, halloffame=hof,  verbose=self.verbose_flag)
        self.hof = hof
        self.segment = hof
        return self

    def predict(self, input_features):
        '''
        Function to predict via the best model. Takes as input features for prediction.
        input_features: The set of features for which you wish to predict the output.
        '''
        if(self.model_type == 'regression'):
            predict_1 = []
            indiv = self.hof[0]
            for i in range(0,len(indiv)):
                predict_1.append(self.get_pre(indiv[i], input_features))
            result = [];
            for i in range(0,len(indiv)):
                if(i==0):
                    result = predict_1[0];
                else:
                    result = result + predict_1[i];
            final_result = result/len(indiv)
        elif(self.model_type == 'classification'):
            predict_list = []
            indiv = self.hof[0]
            for i in range(0,len(indiv)):
                predict_list.append(self.get_pre(indiv[i], input_features))
            final_pred = np.array([Counter(instance_pred).most_common(1)[0][0] for instance_pred in zip(*predict_list)])
            final_result = final_pred
        return final_result

    def get_pre(self, chrom, input_features):
        feat_chrom = list(chrom.X.columns.values)
        test_feat = input_features[feat_chrom]
        mod = chrom.estimator
        predicted = mod.predict(test_feat)
        return predicted