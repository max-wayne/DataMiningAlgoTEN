# -*-coding:utf-8-*-
# @Created at: 2019-04-26 10:27
# @Author: Wayne

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def main():
    # load data
    df = pd.read_csv('./rec.csv', sep='\t')

    # pre-process
    X = df.values[:, :-1]
    y = df.values[:, -1]
    
    # split trainSet and testSet
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # training
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(X_train, y_train)

    # feature importance
    print(clf.feature_importances_)

    # predict
    y_pred = clf.predict(X_test)
    print(np.mean(y_pred == y_test))

    # evaluate
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()





