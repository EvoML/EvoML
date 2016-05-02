import random
from random import randint
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error
import math
from sklearn.cross_validation import train_test_split
from evaluators import evalOneMax
from evaluators import evalOneMax2
from mutators import mutate_feat
from util import compare_hof
from sklearn.base import BaseEstimator, RegressorMixin
from EstimatorGene import EstimatorGene

class Feature_Stacker(BaseEstimator,RegressorMixin):

    def __init__(self, test_size=0.20,N_population=30,N_individual=5,featMin=1,featMax=None, indpb=0.05, ngen = 50, mutpb = 0.40, cxpb = 0.50, base_estimator=linear_model.LinearRegression(), crossover_func = tools.cxTwoPoint, test_frac=0.30, test_frac_flag = False):
        """
        test_size
        """
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
        self.featMin = featMin
        self.featMax = featMax        

    def get_indiv_sample(self,data,output,base_estimator):
        if self.featMax is None:
            self.featMax = data.shape[1]-1
        feat_name = list(data.columns.values)
        feat_count = (randint(self.featMin,self.featMax))
        ind = random.sample(range(0, self.featMax), feat_count)
        new_feat = []
        for i in range(0,len(ind)):
            new_feat.append(feat_name[ind[i]])
        # The below line is just to create a GT but while using bagging.
        tmp_data = data[new_feat].sample(frac=1, replace=True)
        # Just remove the line below if it doesn't run properly.
        output_new = output.loc[list(tmp_data.index)]
        return EstimatorGene(tmp_data,output_new,[],[],base_estimator)

    def fit(self, X, y):
        #ToDo: Check that the columns or the feature names are not same
        #ToDo: All other general sanity checks are also to be made.
        #ToDo: make all the parameters in the init function

        input_feat = list(X.columns.values);
        creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", self.get_indiv_sample, data=X_train, output=y_train, base_estimator=self.base_estimator)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=self.N_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evalOneMax,x_te = X_test, y_te = y_test, test_frac = self.test_frac, test_frac_flag = self.test_frac_flag)
        toolbox.register("mate", self.crossover_func)
        toolbox.register("mutate", mutate_feat, indpb=self.indpb,input_fe = input_feat, X_tr = X_train)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        pop = toolbox.population(n=self.N_population)
        hof = tools.HallOfFame(1, similar=compare_hof);
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        self.pop, self.logbook = algorithms.eaSimple(pop, toolbox, cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.ngen, stats=stats, halloffame=hof,  verbose=False)
        self.hof = hof
        #return pop, logbook, hof
        return self

    def predict(self, input_features):
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
        return final_result

    def get_pre(self, chrom, input_features):
        feat_chrom = list(chrom.X.columns.values)
        test_feat = input_features[feat_chrom]
        mod = chrom.estimator
        #mod = chrom.base_estimator
        #mod.fit(chrom.X, chrom.y)
        predicted = mod.predict(test_feat)
        return predicted


# This is the OOB based Feature Stacker
class Feature_Stacker2(BaseEstimator,RegressorMixin):

    def __init__(self, test_size=0.30,N_population=40,N_individual=5,featMin=1,featMax=None, indpb=0.05, ngen = 50, mutpb = 0.40, cxpb = 0.50, base_estimator=linear_model.LinearRegression(), crossover_func = tools.cxTwoPoint, test_frac=0.30, test_frac_flag = False):
        """
        test_size
        """
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
        self.featMin = featMin
        self.featMax = featMax

    def get_indiv_sample_bag(self,data,output,base_estimator):        
        if self.featMax is None:
            self.featMax = data.shape[1]-1
        feat_name = list(data.columns.values)
        feat_count = (randint(self.featMin,self.featMax))
        ind = random.sample(range(0, self.featMax), feat_count)
        new_feat = []
        for i in range(0,len(ind)):
            new_feat.append(feat_name[ind[i]])
        new_feat_set = data[new_feat]
        new_feat_set = new_feat_set.sample(frac=1,replace=True)
        output_new = output.loc[list(new_feat_set.index)]
        #print len(set(data.index))
        #print len(set(new_feat_set.index))
        #print len(set(data.index)-set(new_feat_set.index))
        return EstimatorGene(new_feat_set,output_new,[],[],base_estimator)

    def fit(self, X, y):
        #ToDo: Check that the columns or the feature names are not same
        #ToDo: All other general sanity checks are also to be made.
        #ToDo: make all the parameters in the init function
        input_feat = list(X.columns.values);
        creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.X = X
        self.y = y
        #self.unseen_y = unseen_y
        #self.unseen_X = unseen_X
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", self.get_indiv_sample_bag, data=X, output=y, base_estimator=self.base_estimator)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=self.N_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evalOneMax2,X_f=X, y_f=y)
        toolbox.register("mate", self.crossover_func)
        toolbox.register("mutate", mutate_feat, indpb=self.indpb,input_fe = input_feat, X_tr = X)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        pop = toolbox.population(n=self.N_population)
        hof = tools.HallOfFame(1, similar=compare_hof);
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        self.pop, self.logbook = algorithms.eaSimple(pop, toolbox, cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.ngen, stats=stats, halloffame=hof,  verbose=False)
        self.hof = hof
        #return pop, logbook, hof
        return self

    def predict(self, input_features):
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
        return final_result

    def get_pre(self, chrom, input_features):
        feat_chrom = list(chrom.X.columns.values)
        test_feat = input_features[feat_chrom]
        mod = chrom.estimator
        #mod = chrom.base_estimator
        #mod.fit(chrom.X, chrom.y)
        predicted = mod.predict(test_feat)
        return predicted