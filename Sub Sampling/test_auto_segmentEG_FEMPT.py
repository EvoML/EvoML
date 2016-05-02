from auto_segment_FEMPT import BasicSegmenterEG_FEMPT
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
import sklearn.datasets as sk_data


def main(X, y, test_size = 0.1):

    # boston = load_boston()
    # X = pd.DataFrame(boston.data)
    # y = pd.DataFrame(boston.target)

    base_estimator = DecisionTreeRegressor(max_depth = 5)
    # base_estimator = LinearRegression()




    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print X_train.shape

    # bench =  BaggingRegressor(base_estimator = base_estimator, n_estimators = 10, max_samples = 0.67, oob_score = True).fit(X_train, y_train)
    # print bench.score(X_test, y_test)
    # print mean_squared_error(bench.predict(X_test), y_test)



    clf = BasicSegmenterEG_FEMPT(ngen=30, init_sample_percentage = 1, n_votes=10, n = 10, base_estimator = base_estimator, unseen_x = X_test, unseen_y = y_test)
    clf.fit(X_train, y_train)
    print clf.score(X_test,y_test)
    y = clf.predict(X_test)
    print mean_squared_error(y, y_test)
    print y.shape
    print type(y)

    return clf, X_test, y_test



from operator import attrgetter


# def get_hof_fitness_unseen(clf, unseen_x, unseen_y, run = None):
    
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

#     # First population is actually the last one being repeated.
#     pops = pops[1:]
#     gen_one_first_ind_fitness = None

#     fs = []
#     uns = []
#     for gen, pop in enumerate(pops):
#         if pop == None:
#             continue
#         best_ind_pop = max(pop, key=attrgetter("fitness"))

#         # get performance on unseen from this.   
#         unseen_mse = eval_unseen_per_gen(best_ind_pop, unseen_x, unseen_y, clf)
#         fitness = best_ind_pop.fitness.values[0]

#         fs.append(fitness) 
#         uns.append(unseen_mse)

#         if gen == 0:
#             ## Saving the first individual of the first generations performance. 
#             ## This will be the benchmark to compare random bagging. 

#             gen_one_first_ind_unseen = eval_unseen_per_gen(pop[0], unseen_x, unseen_y, clf)
#             gen_one_first_ind_fitness = pop[0].fitness.values[0]

#     import matplotlib.pyplot as plt
#     # plt.scatter(uns, fs)
#     gens = range(0,len(pops))
#     uns = [np.sqrt(u) for u in uns]
    
#     corrs =  np.corrcoef(fs, uns)[0][1]
    
#     plt.scatter(gens, fs, c = 'red')
#     plt.scatter(gens, uns, c = 'blue')

#     if run != None:
#         plt.savefig('./Pictures/FEMPT_boston/Changing fitness with gens - run {0}'.format(run))
#         plt.close()
#     else:
#         plt.show()


#     hof_prediction = eval_unseen_per_gen(clf.segments_, unseen_x, unseen_y, clf)
#     hof_fitness = clf.segments_.fitness.values[0]


#     ## analysis of segments. 
#     segments = []
#     number_of_uniques = []
#     for eg in clf.segments_:
#         segments.append(eg.get_data().shape[0])
#         number_of_uniques.append(len(set(eg.get_data().index)))

#     hof_shape_mean = np.mean(segments)
#     hof_shape_var = np.var(segments)
#     number_of_uniques_mean = np.mean(number_of_uniques)
#     number_of_uniques_var = np.var(number_of_uniques)



#     print "---Report---\n"
#     print segments
#     print number_of_uniques
#     print corrs, hof_fitness, hof_prediction,  gen_one_first_ind_unseen, gen_one_first_ind_fitness

#     return corrs, hof_fitness**2, hof_prediction,  gen_one_first_ind_unseen, gen_one_first_ind_fitness**2, hof_shape_mean, hof_shape_var, number_of_uniques_mean, number_of_uniques_var

# if __name__ == '__main__':
#     clf, x, y = main()

#     get_hof_fitness_unseen(clf, x, y)    

# def exp():
#     corr_coefs = []
#     hof_fitnesses = []
#     hof_predictions = []
#     gen_firsts_predictions = []
#     gen_firsts_fitnesses = []

#     hof_shape_means = []
#     hof_shape_vars = []
#     number_of_uniques_means = []
#     number_of_uniques_vars = []


#     for i in range(0,100):
#         print "Run #{0}".format(i)
#         clf, x, y = main()
#         corr, hof_fitness, hof_prediction, gen_one_first_ind_unseen, gen_one_first_ind_fitness, hof_shape_mean, hof_shape_var, number_of_uniques_mean, number_of_uniques_var = get_hof_fitness_unseen(clf, x, y, run = i)
        
#         ### Do some analysis on HOF and get metafeatures.
            

#         corr_coefs.append(corr)
        
#         hof_predictions.append(hof_prediction)
#         hof_fitnesses.append(hof_fitness)

#         gen_firsts_fitnesses.append(gen_one_first_ind_fitness)
#         gen_firsts_predictions.append(gen_one_first_ind_unseen)

#         hof_shape_means.append(hof_shape_mean)
#         hof_shape_vars.append(hof_shape_var)
#         number_of_uniques_means.append(number_of_uniques_mean)
#         number_of_uniques_vars.append(number_of_uniques_var)

#     pd.DataFrame(
#         {
#         'coeff':corr_coefs,
#         'hof_fitnesses': hof_fitnesses,
#         'hof_predictions': hof_predictions,
#         'gen_firsts_fitnesses': gen_firsts_fitnesses,
#         'gen_firsts_predictions': gen_firsts_predictions,
#         'hof_shape_means': hof_shape_means,
#         'hof_shape_vars': hof_shape_vars,
#         'number_of_uniques_means': number_of_uniques_means,
#         'number_of_uniques_vars' : number_of_uniques_vars

#         }).to_csv('FEMPT_experiment_runs_boston.csv')

def get_hof_fitness_unseen(clf, unseen_x, unseen_y, run = None, dataset_name = ''):
    
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
    # Replace this line for the full comment
    # pops = pops[1:]
    # pops = pops[0:]
    # print len(pops)
    # print len(pops[0])
    # print len(pops[0][0])
    # gen_one_first_ind_fitness = None

    fs = []
    uns = []

    gen_one_first_ind_unseen = eval_unseen_per_gen(pops[0][0], unseen_x, unseen_y, clf)
    gen_one_first_ind_fitness = pops[0][0].fitness.values[0]

    ## Uncomment these lines for the full experiment.
    # for gen, pop in enumerate(pops):
    #     if pop == None:
    #         continue
    #     best_ind_pop = max(pop, key=attrgetter("fitness"))

    #     # get performance on unseen from this.   
    #     unseen_mse = eval_unseen_per_gen(best_ind_pop, unseen_x, unseen_y, clf)
    #     fitness = best_ind_pop.fitness.values[0]

    #     fs.append(fitness) 
    #     uns.append(unseen_mse)

    #     if gen == 0:
    #         ## Saving the first individual of the first generations performance. 
    #         ## This will be the benchmark to compare random bagging. 

    #         gen_one_first_ind_unseen = eval_unseen_per_gen(pop[0], unseen_x, unseen_y, clf)
    #         gen_one_first_ind_fitness = pop[0].fitness.values[0]

    # import matplotlib.pyplot as plt
    # # plt.scatter(uns, fs)
    # gens = range(0,len(pops))
    # uns = [np.sqrt(u) for u in uns]
    
    # corrs =  np.corrcoef(fs, uns)[0][1]
    
    # plt.scatter(gens, fs, c = 'red')
    # plt.scatter(gens, uns, c = 'blue')

    # if run != None:
    #     plt.savefig('./Pictures/FEMPT_{0}/Changing fitness with gens - run {1}'.format(dataset_name, run))
    #     plt.close()
    # else:
    #     plt.show()

    corrs = 0

    hof_prediction = eval_unseen_per_gen(clf.segments_, unseen_x, unseen_y, clf)
    hof_fitness = clf.segments_.fitness.values[0]


    ## analysis of segments. 
    segments = []
    number_of_uniques = []
    for eg in clf.segments_:
        segments.append(eg.get_data().shape[0])
        number_of_uniques.append(len(set(eg.get_data().index)))

    hof_shape_mean = np.mean(segments)
    hof_shape_var = np.var(segments)
    number_of_uniques_mean = np.mean(number_of_uniques)
    number_of_uniques_var = np.var(number_of_uniques)



    print "---Report---\n"
    print segments
    print number_of_uniques
    print corrs, hof_fitness, hof_prediction,  gen_one_first_ind_unseen, gen_one_first_ind_fitness

    return corrs, hof_fitness**2, hof_prediction,  gen_one_first_ind_unseen, gen_one_first_ind_fitness**2, hof_shape_mean, hof_shape_var, number_of_uniques_mean, number_of_uniques_var

if __name__ == '__main__':
    clf, x, y = main()

    get_hof_fitness_unseen(clf, x, y)    

def exp(dataset_name, runs):
    base_address = "./Notebooks/Datasets/"
    dataSets = {}   
    dataSets_temp = []
    data_names = ['servo','abalone','ozone']
    for i in range(0,len(data_names)):
        dataSets_temp.append(pd.read_csv(base_address+data_names[i]+'.csv'))
        temp_data = pd.read_csv(base_address+data_names[i]+'.csv')
        temp_data = pd.get_dummies(temp_data)
        temp_output = pd.DataFrame(temp_data['output'])
        temp_data.drop('output',axis=1,inplace=True)
        temp_data = pd.concat([temp_data, temp_output], axis=1)
        dataSets[data_names[i]] = temp_data

    # Friedman Datasets
    data_names.append('fd_1')
    fd_1 = sk_data.make_friedman1(n_samples=2200, n_features=10, random_state=None)
    features = pd.DataFrame(fd_1[0])
    features.columns = ['feat_0','feat_1','feat_2','feat_3','feat_4','feat_5','feat_6','feat_7','feat_8','feat_9']
    output = pd.DataFrame(fd_1[1])
    output.columns = ['output']
    fd_1 = pd.concat([features, output],axis=1)

    data_names.append('fd_2')
    fd_2 = sk_data.make_friedman2(n_samples=2200, random_state=None)
    features = pd.DataFrame(fd_2[0])
    features.columns = ['feat_0','feat_1','feat_2','feat_3']
    output = pd.DataFrame(fd_2[1])
    output.columns = ['output']
    fd_2 = pd.concat([features, output],axis=1)

    data_names.append('fd_3')
    fd_3 = sk_data.make_friedman3(n_samples=2200, random_state=None)
    features = pd.DataFrame(fd_3[0])
    features.columns = ['feat_0','feat_1','feat_2','feat_3']
    output = pd.DataFrame(fd_3[1])
    output.columns = ['output']
    fd_3 = pd.concat([features, output],axis=1)
 
    dataSets['fd_1'] = fd_1
    dataSets['fd_2'] = fd_2
    dataSets['fd_3'] = fd_3


    if dataset_name in ['fd_1', 'fd_2', 'fd_3']:
        test_size = 0.909
    elif dataset_name == 'abalone':
        test_size = 0.25
    else:
        test_size = 0.1


    data_ = dataSets[dataset_name]
    y = data_['output']
    data_.drop('output',axis=1,inplace=True)
    X = data_



    corr_coefs = []
    hof_fitnesses = []
    hof_predictions = []
    gen_firsts_predictions = []
    gen_firsts_fitnesses = []

    hof_shape_means = []
    hof_shape_vars = []
    number_of_uniques_means = []
    number_of_uniques_vars = []




    for i in range(0,runs):
        print "Run #{0}".format(i)
        clf, x_test, y_test = main(X, y, test_size = test_size)
        corr, hof_fitness, hof_prediction, gen_one_first_ind_unseen, gen_one_first_ind_fitness, hof_shape_mean, hof_shape_var, number_of_uniques_mean, number_of_uniques_var = get_hof_fitness_unseen(clf, x_test, y_test, run = i, dataset_name = dataset_name)
        
        ### Do some analysis on HOF and get metafeatures.
            
        corr_coefs.append(corr)
        
        hof_predictions.append(hof_prediction)
        hof_fitnesses.append(hof_fitness)

        gen_firsts_fitnesses.append(gen_one_first_ind_fitness)
        gen_firsts_predictions.append(gen_one_first_ind_unseen)

        hof_shape_means.append(hof_shape_mean)
        hof_shape_vars.append(hof_shape_var)
        number_of_uniques_means.append(number_of_uniques_mean)
        number_of_uniques_vars.append(number_of_uniques_var)

    pd.DataFrame(
        {
        'coeff':corr_coefs,
        'hof_fitnesses': hof_fitnesses,
        'hof_predictions': hof_predictions,
        'gen_firsts_fitnesses': gen_firsts_fitnesses,
        'gen_firsts_predictions': gen_firsts_predictions,
        'hof_shape_means': hof_shape_means,
        'hof_shape_vars': hof_shape_vars,
        'number_of_uniques_means': number_of_uniques_means,
        'number_of_uniques_vars' : number_of_uniques_vars

                }).to_csv('FEMPT_experiment_runs_{0}.csv'.format(dataset_name))

