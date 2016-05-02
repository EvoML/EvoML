from auto_segment_FEGO import BasicSegmenterEG_FEGO
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston


def main():
    df = pd.read_csv('ameo_numerical.csv')

    x_columns = list(set(df.columns) - set(['Salary']))
    y_column = 'Salary'
    X = df[x_columns]
    y = df[y_column]

    boston = load_boston()
    X = pd.DataFrame(boston.data)
    y = pd.DataFrame(boston.target)




    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 12)
    print X_train.shape

    print LinearRegression().fit(X_train, y_train).score(X_test, y_test)

    clf = BasicSegmenterEG_FEGO(ngen=100, init_sample_percentage = 0.5)
    clf.fit(X_train, y_train)
    print clf.score(X_test,y_test)
    y = clf.predict(X_test)
    print mean_squared_error(y, y_test)
    print y.shape
    print type(y)




if __name__ == '__main__':
    main()