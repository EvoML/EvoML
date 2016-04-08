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
from mutators import mutate_feat
from util import compare_hof
from sklearn.base import BaseEstimator, RegressorMixin

class Feature_Stacker(BaseEstimator,RegressorMixin):

    def __init__(self, partition=0.30,population_cnt=40,individual_cnt=5,N_feat_min=5,N_feat_max=21, indpb=0.70, ngen = 50, mutpb = 0.40, cxpb = 0.50, base_estimator=linear_model.LinearRegression(), crossover_func = tools.cxTwoPoint):
        # partition - text_size
        # population_cnt - N_population
        # min max xlim[]
        self.partition = partition
        self.population_cnt = population_cnt
        self.individual_cnt = individual_cnt
        self.N_feat_max = N_feat_max
        self.N_feat_min = N_feat_min
        self.indpb = indpb
        self.ngen = ngen
        self.mutpb = mutpb
        self.cxpb = cxpb
        self.base_estimator = base_estimator
        self.crossover_func = crossover_func

    def get_indiv_sample(self,data):
        feat_name = list(data.columns.values)
        feat_count = (randint(self.N_feat_min,self.N_feat_max))
        ind = random.sample(range(0, self.N_feat_max), feat_count)
        new_feat = []
        for i in range(0,len(ind)):
            new_feat.append(feat_name[ind[i]])
        return data[new_feat]

    def fit(self, X, y):
        #ToDo: Check that the columns or the feature names are not same
        #ToDo: All other general sanity checks are also to be made.
        #ToDo: make all the parameters in the init function

        input_feat = list(X.columns.values);
        creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.partition, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", self.get_indiv_sample, data=X_train)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=self.individual_cnt)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evalOneMax, y_tr = y_train, x_te = X_test, y_te = y_test, base_estimator = self.base_estimator);
        toolbox.register("mate", self.crossover_func)
        toolbox.register("mutate", mutate_feat, indpb=self.indpb,input_fe = input_feat, X_tr = X_train)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        pop = toolbox.population(n=self.population_cnt)
        hof = tools.HallOfFame(1, similar=compare_hof);
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.ngen, stats=stats, halloffame=hof,  verbose=True)
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
        feat_chrom = list(chrom.columns.values)
        test_feat = input_features[feat_chrom]
        mod = linear_model.LinearRegression()
        mod.fit(chrom, self.y_train)
        predicted = mod.predict(test_feat)
        return predicted