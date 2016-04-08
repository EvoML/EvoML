import random
from random import randint
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error
import math

def mutate_feat(individual, indpb, input_fe, X_tr):
    for i in range(0,len(individual)):
        chrom = individual[i]
        if(random.random()<indpb):
            k = random.random()
            if(k<0.33):
                if(chrom.shape[1]>8):
                    ind = random.randint(0,chrom.shape[1]-1)
                    final_feat = list(chrom.columns.values)
                    final_feat.remove(final_feat[ind])
                    chrom = chrom[final_feat]
            if(k<0.50):
                present_feat = list(chrom.columns.values)
                final_feat = list(set(input_fe)-set(present_feat))
                if(len(final_feat)>0):
                    ind = random.randint(0,len(final_feat)-1)
                    present_feat.append(final_feat[ind])
                    chrom = X_tr[present_feat]
            else:
                present_feat = list(chrom.columns.values)
                ind = random.randint(0,len(present_feat)-1)
                final_feat = list(set(input_fe)-set(present_feat))
                if(len(final_feat)>0):
                    ind1 = random.randint(0,len(final_feat)-1)
                    present_feat[ind] = final_feat[ind1]
                    chrom = X_tr[present_feat]
        else:
            # nothing do here as of now
            do_nothing = 2;
        return (individual,)