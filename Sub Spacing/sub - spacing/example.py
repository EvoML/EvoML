from auto_feature import Feature_Stacker
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv('ameo_numerical.csv')
    input_feat = ['Gender', '10percentage', '12graduation',
         '12percentage', 'CollegeTier', 'collegeGPA', 'CollegeCityTier',
         'GraduationYear', 'English', 'Logical', 'Quant', 'Domain',
         'conscientiousness', 'agreeableness', 'extraversion', 'nueroticism',
         'openess_to_experience', 'Degree_B.Tech/B.E.',
         'Degree_M.Sc. (Tech.)', 'Degree_M.Tech./M.E.', 'Degree_MCA']
    output_feat = ['Salary']
    features = data[input_feat]
    output = data[output_feat]
    FS = Feature_Stacker(ngen=20)
    FS.fit(features,output)
    #print FS.get_params()
    final_result = FS.predict(features)
    print FS.score(features, output)
    #pop, log, hof = fit(self,features,output)
    #print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
    #print("Best individual fitness: %s" % (hof[0].fitness))