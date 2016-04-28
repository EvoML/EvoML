from auto_feature import Feature_Stacker
from auto_feature import Feature_Stacker2
import pandas as pd
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn.tree import DecisionTreeRegressor


if __name__ == "__main__":
    data = pd.read_csv('ameo_numerical.csv')
    data_submission = pd.read_csv('ameo_numerical_test.csv');
    data_submission.Domain = data_submission.Domain.fillna(data_submission.Domain.mean())
    input_feat = ['Gender', '10percentage', '12graduation',
         '12percentage', 'CollegeTier', 'collegeGPA', 'CollegeCityTier',
         'GraduationYear', 'English', 'Logical', 'Quant', 'Domain',
         'conscientiousness', 'agreeableness', 'extraversion', 'nueroticism',
         'openess_to_experience', 'Degree_B.Tech/B.E.',
         'Degree_M.Sc. (Tech.)', 'Degree_M.Tech./M.E.', 'Degree_MCA']
    output_feat = ['Salary']
    features = data[input_feat]
    output = data[output_feat]
    #FS = Feature_Stacker(ngen=20,base_estimator=DecisionTreeRegressor(max_depth=2),N_individual=50)
    FS = Feature_Stacker2(ngen=7)
    '''
    FS.fit(features,output)
    final_result = FS.predict(features)
    print FS.score(features, output)
    test_features = data_submission[input_feat]
    test_result = FS.predict(test_features)
    test_result = pd.DataFrame(test_result)
    test_result.to_csv('test_result.csv')
    '''
    #parameters = {'indpb':np.arange(0.1,0.6,0.1).tolist(), 'mutpb':np.arange(0.1,0.7,0.1).tolist(),'cxpb':np.arange(0.1,0.7,0.1).tolist(),'N_individual':np.arange(4,10,1).tolist()}
    #grid_model = GridSearchCV(FS, parameters, verbose=True)
    #grid_model = RandomizedSearchCV(FS, parameters, verbose=True, scoring="mean_squared_error")
    #grid_model.fit(features, output)
    #final_result = grid_model.predict(features)
    grid_model = FS
    grid_model.fit(features, output)
    final_result = grid_model.predict(features)
    print grid_model.score(features,output)
    #print grid_model.best_params_
    print grid_model.get_params
    test_features = data_submission[input_feat]
    test_result = grid_model.predict(test_features)
    test_result = pd.DataFrame(test_result)
    #test_result.to_csv('test_result_1.csv')
    #pop, log, hof = fit(self,features,output)
    #print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
    #print("Best individual fitness: %s" % (hof[0].fitness))"""