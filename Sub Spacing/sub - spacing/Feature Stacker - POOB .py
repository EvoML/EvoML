
# coding: utf-8

# In[6]:

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
boston = load_boston()
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


# In[3]:

def get_result(model_,data_, frac):
    model = clone(model_)
    output = data_['output']
    data_.drop('output',axis=1,inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(data_, output, test_size=frac)
    model.fit(X_train,y_train)
    test_result = model.predict(X_test)
    return mean_squared_error(y_test,test_result), test_result, model


# In[4]:

def get_result_iter(data, N_iterations, frac, models):
    #get_result(FS,temp_data)
    model_list = []
    Result_mses = []
    for i in range(0,len(models)):
        model_iter_results = []
        for j in range(0,N_iterations):
            temp_data = data.copy(deep=True)
            temp_rs, temp_pred, model = get_result(models[i],temp_data, frac)
            #print model.best_params_
            model_iter_results.append(temp_rs)
            model_list.append(model)
        Result_mses.append(sum(model_iter_results)/len(model_iter_results))
    return Result_mses, model_list


# In[5]:

Res_all_ds = []
model_f = []
for i in range(0,len(dataSets)):
    t_data = dataSets[i].copy(deep=True)
    N_iterations = 1
    g_frac = 0.10
    #parameters = {'indpb':np.arange(0.1,0.6,0.2).tolist(), 'mutpb':np.arange(0.1,0.7,0.2).tolist(),
                  #'cxpb':np.arange(0.1,0.7,0.2).tolist(),'N_individual':np.arange(4,15,2).tolist(),
                  #'test_frac':np.arange(0.1,1,0.2)}
    parameters = {'indpb':np.arange(0.1,0.6,0.2).tolist(),'N_individual':[5,10,20,25,35,50]}
    FS = auto_feature.Feature_Stacker2(ngen=20,cxpb = 0.6, mutpb = 0.4)
    #grid_model = RandomizedSearchCV(FS, parameters, verbose=True, scoring="mean_squared_error",n_iter=100)
    grid_model = GridSearchCV(FS, parameters, verbose=True, scoring="mean_squared_error")
    g_models = []
    g_models.append(grid_model)
    #g_models.append(linear_model.LinearRegression())
    #g_models.append(linear_model.LassoCV(n_alphas=100))
    Result_t, model_list = get_result_iter(t_data, N_iterations, g_frac, g_models)
    Res_all_ds.append(Result_t)
    model_f.append(model_list)
dataSet_names = ['Servo','Abalone','Ozone','Boston-housing']
model_names = ['Feature Stacker-CV']
Result_test_bench = pd.DataFrame(Res_all_ds)
Result_test_bench.columns = model_names
Result_test_bench.set_index([dataSet_names],inplace=True)
Result_test_bench


# In[12]:

import sklearn.datasets as sk_data


# In[35]:

# Friedman Datasets
fd_1 = sk_data.make_friedman1(n_samples=2200, n_features=10, random_state=None)
features = pd.DataFrame(fd_1[0])
features.columns = ['feat_0','feat_1','feat_2','feat_3','feat_4','feat_5','feat_6','feat_7','feat_8','feat_9']
output = pd.DataFrame(fd_1[1])
output.columns = ['output']
fd_1 = pd.concat([features, output],axis=1)

fd_2 = sk_data.make_friedman2(n_samples=1200, random_state=None)
features = pd.DataFrame(fd_2[0])
features.columns = ['feat_0','feat_1','feat_2','feat_3']
output = pd.DataFrame(fd_2[1])
output.columns = ['output']
fd_2 = pd.concat([features, output],axis=1)

fd_3 = sk_data.make_friedman3(n_samples=200, random_state=None)
features = pd.DataFrame(fd_3[0])
features.columns = ['feat_0','feat_1','feat_2','feat_3']
output = pd.DataFrame(fd_3[1])
output.columns = ['output']
fd_3 = pd.concat([features, output],axis=1)

dataSets.append(fd_1)
dataSets.append(fd_2)
dataSets.append(fd_3)


# In[48]:

model_f[0][0].best_params_


# In[49]:

t_data = dataSets[0].copy(deep=True)
FS1 = auto_feature.Feature_Stacker2(ngen=50,cxpb = 0.6, mutpb = 0.4, N_individual=5, indpb=0.3)
g_models = []
g_models.append(FS1)
Result_t1, model_l1 = get_result_iter(t_data, N_iterations, g_frac, g_models)
print Result_t


# In[50]:

t_data = dataSets[0].copy(deep=True)
FS1 = auto_feature.Feature_Stacker2(ngen=50,cxpb = 0.6, mutpb = 0.4, N_individual=5, indpb=0.3)
g_models = []
g_models.append(FS1)
Result_t1, model_l1 = get_result_iter(t_data, N_iterations, g_frac, g_models)
print Result_t


# In[51]:

dataSets[0]


# In[ ]:




# In[ ]:



