import random
from random import randint
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error
import math
from sklearn.cross_validation import train_test_split

def compare_hof(ind1, ind2):
    return np.all(ind1.fitness==ind2.fitness)