'''
This file contains all the mutation function. Presently, there is only one mutation
function but different mutation functions can be put here and used accordingly. 
The mutation function is called by GA as per mutpb and then indpb helps in deciding 
whether the chromosome inside the individual will be mutated or not.
'''
import random
from random import randint
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error
import math
from .util import EstimatorGene

def mutate_feat(individual, indpb, input_fe, X_tr):
    '''
    The function mutates the chromosome in three ways: 
    1. Insertion
    2. Deletion
    3. Insertion and Deletion
    '''
    for i in range(0,len(individual)):
        chrom = individual[i].X
        if(random.random()<indpb):
            k = random.random()
            if(k<0.33):
                if(chrom.shape[1]>1):
                    ind = random.randint(0,chrom.shape[1]-1)
                    final_feat = list(chrom.columns.values)
                    final_feat.remove(final_feat[ind])
                    chrom = chrom[final_feat]
            if(k<0.66):
                present_feat = list(chrom.columns.values)
                final_feat = list(set(input_fe)-set(present_feat))
                if(len(final_feat)>0):
                    ind = random.randint(0,len(final_feat)-1)
                    present_feat.append(final_feat[ind])
                    ind_row = list(chrom.index)
                    chrom = X_tr.loc[ind_row]
                    chrom = chrom[present_feat]
            else:
                #ToDo Add or remove a set.
                present_feat = list(chrom.columns.values)
                ind = random.randint(0,len(present_feat)-1)
                final_feat = list(set(input_fe)-set(present_feat))
                if(len(final_feat)>0):
                    ind1 = random.randint(0,len(final_feat)-1)
                    present_feat[ind] = final_feat[ind1]
                    ind_row = list(chrom.index)
                    chrom = X_tr.loc[ind_row]
                    chrom = chrom[present_feat]
            individual[i] = EstimatorGene(chrom,individual[i].y, [], [],individual[i].base_estimator)
        else:
            # nothing do here as of now
            do_nothing = 2;
        return (individual,)