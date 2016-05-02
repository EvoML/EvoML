from auto_segment import BasicSegmenter
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

def main():
    df = pd.read_csv('ameo_numerical.csv')
    x_columns = list(set(df.columns) - set(['Salary']))
    y_column = 'Salary'
    X = df[x_columns]
    y = df[y_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

    print LinearRegression().fit(X_train, y_train).score(X_test, y_test)

    clf = BasicSegmenter(ngen=10)
    clf.fit(X_train, y_train)
    print clf.score(X_test,y_test)
    y = clf.predict(X)
    print y.shape
    print type(y)




if __name__ == '__main__':
    main()