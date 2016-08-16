import pandas as pd

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

from .auto_segment_FEMPO import BasicSegmenter_FEMPO

def demo(X = None, y = None, test_size = 0.1):
    
    if X == None:
        boston = load_boston()
        X = pd.DataFrame(boston.data)
        y = pd.DataFrame(boston.target)



    base_estimator = DecisionTreeRegressor(max_depth = 5)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print X_train.shape

    # If you want to compare with BaggingRegressor.
    # bench =  BaggingRegressor(base_estimator = base_estimator, n_estimators = 10, max_samples = 1, oob_score = True).fit(X_train, y_train)
    # print bench.score(X_test, y_test)
    # print mean_squared_error(bench.predict(X_test), y_test)

    clf = BasicSegmenterEG_FEMPO(ngen=30,init_sample_percentage = 1, n_votes=10, n = 10, base_estimator = base_estimator,
        unseen_x = X_test, unseen_y = y_test)
    clf.fit(X_train, y_train)
    print clf.score(X_test,y_test)
    y = clf.predict(X_test)
    print mean_squared_error(y, y_test)
    print y.shape

    return clf, X_test, y_test



