from sklearn.base import clone

class EstimatorGene:
    """ Last column has to be the dependent variable"""
    def __init__(self, X, y, X_test, y_test, base_estimator):
       self.base_estimator = base_estimator
       self.estimator = clone(base_estimator)
       self.estimator.fit(X, y)
       self.X = X
       self.y = y

    def get_data(self):
       return pd.concat([self.X, self.y], axis = 1)