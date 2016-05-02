
# coding: utf-8

# In[1]:

#get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import auto_feature
#dreload(auto_feature)
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.base import clone
import sklearn.datasets as sk_data
import matplotlib.pyplot as plt
boston = load_boston()
from operator import attrgetter
from sklearn.tree import DecisionTreeRegressor
#print(boston.data.shape)


# In[2]:

base_address = "../Datasets/"
dataSets = []
dataSets_temp = []
data_names = ['servo','abalone','ozone']
for i in range(0,len(data_names)):
    dataSets_temp.append(pd.read_csv(base_address+data_names[i]+'.csv'))
    temp_data = pd.read_csv(base_address+data_names[i]+'.csv')
    temp_data = pd.get_dummies(temp_data)
    temp_output = pd.DataFrame(temp_data['output'])
    temp_data.drop('output',axis=1,inplace=True)
    temp_data = pd.concat([temp_data, temp_output], axis=1)
    dataSets.append(temp_data)
# Boston Data
b_feat = pd.DataFrame(boston.data)
b_feat.columns = ['feat_0','feat_1','feat_2','feat_3','feat_4','feat_5','feat_6','feat_7','feat_8','feat_9','feat_10','feat_11','feat_12']
b_target = pd.DataFrame(boston.target)
b_target.columns = ['output']
b_data = pd.concat([b_feat,b_target],axis=1)
dataSets.append(b_data)
# Friedman Datasets
fd_1 = sk_data.make_friedman1(n_samples=2200, n_features=10, random_state=None)
features = pd.DataFrame(fd_1[0])
features.columns = ['feat_0','feat_1','feat_2','feat_3','feat_4','feat_5','feat_6','feat_7','feat_8','feat_9']
output = pd.DataFrame(fd_1[1])
output.columns = ['output']
fd_1 = pd.concat([features, output],axis=1)

fd_2 = sk_data.make_friedman2(n_samples=2200, random_state=None)
features = pd.DataFrame(fd_2[0])
features.columns = ['feat_0','feat_1','feat_2','feat_3']
output = pd.DataFrame(fd_2[1])
output.columns = ['output']
fd_2 = pd.concat([features, output],axis=1)

fd_3 = sk_data.make_friedman3(n_samples=2200, random_state=None)
features = pd.DataFrame(fd_3[0])
features.columns = ['feat_0','feat_1','feat_2','feat_3']
output = pd.DataFrame(fd_3[1])
output.columns = ['output']
fd_3 = pd.concat([features, output],axis=1)

dataSets.append(fd_1)
dataSets.append(fd_2)
dataSets.append(fd_3)

dataSets_frac = [0.10, 0.25, 0.10, 0.10, 0.909, 0.909, 0.909]


# In[3]:

def get_result(model_,data_, frac):
    model = clone(model_)
    output = data_['output']
    data_.drop('output',axis=1,inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(data_, output, test_size=frac)
    model.fit(X_train,y_train)
    model.unseen_x = X_test
    model.unseen_y = y_test
    test_result = model.predict(X_test)
    return mean_squared_error(y_test,test_result), test_result, model
def get_result_iter(data, N_iterations, frac, models):
    #get_result(FS,temp_data)
    model_list = []
    Result_mses = []
    temp_perf_data = []
    for i in range(0,len(models)):
        model_iter_results = []
        for j in range(0,N_iterations):
            temp_data = data.copy(deep=True)
            temp_rs, temp_pred, model = get_result(models[i],temp_data, frac)
            #print model.best_params_
            model_iter_results.append(temp_rs)
            model_list.append(model)
        Result_mses.append(sum(model_iter_results)/len(model_iter_results))
        temp_perf_data.append(model_iter_results)
    return Result_mses, model_list, temp_perf_data


# In[4]:

def test_bench_earlier():
    Res_all_ds = []
    model_f = []
    perf_f = []
    for i in range(0,len(dataSets)):
        t_data = dataSets[i].copy(deep=True)
        N_iterations = 50
        g_frac = dataSets_frac[i]
        regressor = DecisionTreeRegressor(max_depth=5)
        FS = auto_feature.Feature_Stacker2(ngen=40,cxpb = 0.6, mutpb = 0.4, indpb = 0.3, N_individual = 5, base_estimator = regressor)
        g_models = []
        g_models.append(FS)
        Result_t, model_list, model_perf = get_result_iter(t_data, N_iterations, g_frac, g_models)
        perf_f.append(model_perf)
        Res_all_ds.append(Result_t)
        model_f.append(model_list)
    dataSet_names = ['Servo','Abalone','Ozone','Boston-housing', 'FD#1', 'FD#2', 'FD#3']
    model_names = ['Feature Stacker-POOB']
    Result_test_bench = pd.DataFrame(Res_all_ds)
    Result_test_bench.columns = model_names
    Result_test_bench.set_index([dataSet_names],inplace=True)
    Result_test_bench


# In[5]:

def test_bench_main(dataSetChoice):
    Res_all_ds = []
    model_f = []
    perf_f = []
    for i in range(dataSetChoice,dataSetChoice+1):
        t_data = dataSets[i].copy(deep=True)
        N_iterations = 1
        g_frac = 0.10
        regressor = DecisionTreeRegressor(max_depth=5)
        FS = auto_feature.Feature_Stacker(ngen=30,cxpb = 0.6, mutpb = 0.4, indpb = 0.3, N_individual = 5
                                           , base_estimator = regressor, test_frac=1)
        #FS = auto_feature.Feature_Stacker2(ngen=30,cxpb = 0.6, mutpb = 0.4, indpb = 0.3, N_individual = 5
        #                                   , base_estimator = regressor)
        g_models = []
        g_models.append(FS)
        Result_t, model_list, model_perf = get_result_iter(t_data, N_iterations, g_frac, g_models)
        perf_f.append(model_perf)
        Res_all_ds.append(Result_t)
        model_f.append(model_list)
    #dataSet_names = ['Servo','Abalone','Ozone','Boston-housing', 'FD#1', 'FD#2', 'FD#3']
    dataSet_names = ['Servo']
    model_names = ['Feature Stacker-POOB']
    Result_test_bench = pd.DataFrame(Res_all_ds)
    Result_test_bench.columns = model_names
    Result_test_bench.set_index([dataSet_names],inplace=True)
    #print Result_test_bench
    return model_f[0][0]


# In[6]:

def get_hof_fitness_unseen(clf, unseen_x, unseen_y):
    
    def eval_unseen_per_gen(individual, unseen_x, unseen_y, self):
        predict_rmses = []
        predict_vals = []
        final_prediction = []
        ind_f = list(unseen_x.index)
        for i in range(0,len(individual)):
            chromosome = individual[i]
            predict_vals.append(get_predictions(unseen_x, chromosome))
            if(i==0):
                final_prediction = predict_vals[i]
            else:
                final_prediction = final_prediction + predict_vals[i]
            #predict_rmses.append(mean_squared_error(unseen_y, predict_vals[i]))
        final_prediction = final_prediction/len(individual)
        #final_rmse = sum(predict_rmses)/len(predict_rmses)
        final_rmse = mean_squared_error(unseen_y, final_prediction)
        return final_rmse,
    
    def get_predictions(x_te, chrom):
        feat_chrom = list(chrom.X.columns.values)
        test_feat = x_te[feat_chrom]
        mod = chrom.estimator
        predicted = mod.predict(test_feat)
        return predicted
    
    def get_feat_count(all_col, indiv):
        cnt_ = []
        indiv_col = []
        for i in range(0,len(indiv)):
            indiv_col += list(indiv[i].X.columns)
        for i in range(0,len(all_col)):
            cnt_.append(indiv_col.count(all_col[i]))
        return cnt_, len(indiv_col)/len(indiv)
    logbook = clf.logbook
    pops = logbook.select('pop')
    pops = pops[1:]
    fitness_f = []
    unseen_f = []
    first_mse = eval_unseen_per_gen(pops[0][0], unseen_x, unseen_y, clf)
    first_fitness = pops[0][0].fitness.values[0]
    hof_mse = eval_unseen_per_gen(clf.hof[0], unseen_x, unseen_y, clf)
    hof_fitness = clf.hof[0].fitness.values[0]
    feat_cnt, feat_avg_cnt = get_feat_count(list(unseen_x.columns),clf.hof[0])
    for pop in pops:
        best_ind_pop = max(pop, key=attrgetter("fitness"))
        # get performance on unseen from this.   
        unseen_mse = eval_unseen_per_gen(best_ind_pop, unseen_x, unseen_y, clf)
        fitness = best_ind_pop.fitness.values[0]
        #print fitness, unseen_mse[0]
        fitness_f.append(fitness)
        unseen_f.append(unseen_mse[0])
    return fitness_f, unseen_f, first_mse, hof_mse, first_fitness, hof_fitness, feat_cnt, feat_avg_cnt, pop


# In[8]:

#fitness_f, unseen_f = get_hof_fitness_unseen(reg_chk, reg_chk.unseen_x, reg_chk.unseen_y)
g_data = []
g_A = []
for dataSetChoice in range(0,1):
    f_f = []
    u_f = []
    csv_f = []
    name = 'GT-c-datSet-'+str(dataSetChoice)
    for i in range(0,5):
        A = test_bench_main(dataSetChoice)
        fitness_f, unseen_f, first_mse, hof_mse, first_fitness, hof_fitness, feat_cnt, feat_avg_cnt, test_pop = get_hof_fitness_unseen(A, A.unseen_x, A.unseen_y)
        f_f.append(fitness_f)
        u_f.append(unseen_f)
        print np.corrcoef(fitness_f, unseen_f)[0][1]
        corr_val = np.corrcoef(fitness_f, unseen_f)[0][1]
        tmp = [corr_val, first_fitness, first_mse[0], hof_fitness, hof_mse[0], feat_avg_cnt]
        tmp += feat_cnt
        csv_f.append(tmp)
        #plt.scatter(np.arange(0,len(u_f[i])),u_f[i],c='blue')
        #plt.scatter(np.arange(0,len(u_f[i])),f_f[i],c='red')
        #plt.savefig('../graphs/fig-'+name+'-'+str(i)+'.png')
        #plt.close()
    data_1 = pd.DataFrame(csv_f)
    data_1.columns = ['corr','first_fitness','first_unseen','hof_fitness','hof_unseen', 'avg_feat_cnt']+list(A.unseen_x.columns)
    #pd.DataFrame(csv_f).to_csv('Result-'+name+'.csv')
    data_1.to_csv('Result-'+name+'.csv')
    g_data.append(data_1)
    g_A.append(A)