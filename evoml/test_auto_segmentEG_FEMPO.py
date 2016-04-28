from auto_segment_FEMPO import BasicSegmenterEG_FEMPO
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

import numpy as np
import pandas as pd
import random
# from mutators import segment_mutator

from evaluators import eval_each_model_PT_KNN_EG

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


def main():
    boston = load_boston()
    X = pd.DataFrame(boston.data)
    y = pd.DataFrame(boston.target)

    base_estimator = DecisionTreeRegressor(max_depth = 5)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    print X_train.shape

    bench =  BaggingRegressor(base_estimator = base_estimator, n_estimators = 10, max_samples = 0.5, oob_score = True).fit(X_train, y_train)
    print bench.score(X_test, y_test)
    print mean_squared_error(bench.predict(X_test), y_test)

    clf = BasicSegmenterEG_FEMPO(ngen=30,init_sample_percentage = 0.5, n_votes=10, n = 10, base_estimator = base_estimator,
        unseen_x = X_test, unseen_y = y_test)
    clf.fit(X_train, y_train)
    print clf.score(X_test,y_test)
    y = clf.predict(X_test)
    print mean_squared_error(y, y_test)
    print y.shape
    print type(y)

    return clf, X_test, y_test
    

from operator import attrgetter

# def get_hof_fitness_unseen(clf, unseen_x, unseen_y):
    
#     def eval_unseen_per_gen(ind, unseen_x, unseen_y, self):

#         """
#         Unseen is taken from init params and is a complete  
#         """
#         ensembles = []
#         centroids = []

#         X = unseen_x.copy()
#         # scaling using the mean and std of the original train data.
#         X = (X - self._X_mean)/self._X_std
#         for eg_ in ind:
#             df_ = eg_.get_data()
#             ensembles.append(eg_.estimator)
#             centroids.append(centroid_df(df_.iloc[:,0:-1]))
        
#         y_preds_ensemble = []
#         ensembles = np.array(ensembles)
#         for row in X.values:
#             distances = np.array([distance(row, centroid) for centroid in centroids])
#             model_ixs = distances.argsort()[:self.n_votes]
#             models = ensembles[model_ixs]
#             y_pred = np.mean([mdl.predict(row)[0] for mdl in models])
#             y_preds_ensemble.append(y_pred)
#             ## MSE
        
#         return mean_squared_error(y_preds_ensemble, unseen_y)


#     logbook = clf.log
#     pops = logbook.select('pop')

#     uns = []
#     fs = []
#     for pop in pops:
#         if pop == None:
#             continue
#         best_ind_pop = max(pop, key=attrgetter("fitness"))

#         # get performance on unseen from this.   
#         unseen_mse = eval_unseen_per_gen(best_ind_pop, unseen_x, unseen_y, clf)
#         fitness = best_ind_pop.fitness.values[0]

#         fs.append(fitness) 
#         uns.append(unseen_mse)

#     import matplotlib.pyplot as plt
#     # plt.scatter(uns, fs)
#     gens = range(0,len(pops))
#     plt.scatter(gens, fs, c = 'red')
#     plt.scatter(gens, uns, c = 'blue')


#     print pd.DataFrame({'uns' : uns, 'fs' : fs}).corr()
#     plt.show()

# if __name__ == '__main__':
#     clf, x, y = main()

#     get_hof_fitness_unseen(clf, x, y)



def get_hof_fitness_unseen(clf, unseen_x, unseen_y, run = None):
    
    def eval_unseen_per_gen(ind, unseen_x, unseen_y, self):

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


    logbook = clf.log
    pops = logbook.select('pop')

    # First population is actually the last one being repeated.
    pops = pops[1:]
    gen_one_first_ind_fitness = None

    fs = []
    uns = []
    for gen, pop in enumerate(pops):
        if pop == None:
            continue
        best_ind_pop = max(pop, key=attrgetter("fitness"))

        # get performance on unseen from this.   
        unseen_mse = eval_unseen_per_gen(best_ind_pop, unseen_x, unseen_y, clf)
        fitness = best_ind_pop.fitness.values[0]

        fs.append(fitness) 
        uns.append(unseen_mse)

        if gen == 0:
            ## Saving the first individual of the first generations performance. 
            ## This will be the benchmark to compare random bagging. 

            gen_one_first_ind_unseen = eval_unseen_per_gen(pop[0], unseen_x, unseen_y, clf)
            gen_one_first_ind_fitness = pop[0].fitness.values[0]

    import matplotlib.pyplot as plt
    # plt.scatter(uns, fs)
    gens = range(0,len(pops))
    uns = [np.sqrt(u) for u in uns]
    
    corrs =  np.corrcoef(fs, uns)[0][1]
    
    plt.scatter(gens, fs, c = 'red')
    plt.scatter(gens, uns, c = 'blue')

    if run != None:
        plt.savefig('./Pictures/FEMPO/Changing fitness with gens - run {0}'.format(run))
        plt.close()
    else:
        plt.show()


    hof_prediction = eval_unseen_per_gen(clf.segments_, unseen_x, unseen_y, clf)
    hof_fitness = clf.segments_.fitness.values[0]

    print "---Report---\n"
    print corrs, hof_fitness, hof_prediction,  gen_one_first_ind_unseen, gen_one_first_ind_fitness

    return corrs, hof_fitness**2, hof_prediction,  gen_one_first_ind_unseen, gen_one_first_ind_fitness**2

if __name__ == '__main__':
    clf, x, y = main()

    get_hof_fitness_unseen(clf, x, y)    

def exp():
    corr_coefs = []
    hof_fitnesses = []
    hof_predictions = []
    gen_firsts_predictions = []
    gen_firsts_fitnesses = []

    for i in range(0,200):
        print "Run #{0}".format(i)
        clf, x, y = main()
        corr, hof_fitness, hof_prediction, gen_one_first_ind_unseen, gen_one_first_ind_fitness = get_hof_fitness_unseen(clf, x, y, run = i)
        corr_coefs.append(corr)
        
        hof_predictions.append(hof_prediction)
        hof_fitnesses.append(hof_fitness)

        gen_firsts_fitnesses.append(gen_one_first_ind_fitness)
        gen_firsts_predictions.append(gen_one_first_ind_unseen)

    pd.DataFrame(
        {
        'coeff':corr_coefs,
        'hof_fitnesses': hof_fitnesses,
        'hof_predictions': hof_predictions,
        'gen_firsts_fitnesses': gen_firsts_fitnesses,
        'gen_firsts_predictions': gen_firsts_predictions
                }).to_csv('FEMPO_experiment_runs.csv')
