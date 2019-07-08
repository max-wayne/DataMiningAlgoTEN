# -*-coding:utf-8-*-
# @Created at: 2019-04-22 22:28
# @Author: Wayne
# Ref: https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


def main():
    # load data
    df = pd.read_csv('../k-means/Iris.csv')
    df.drop('Id', axis=1, inplace=True)

    # change categorical data to number 0-2
    df['Species'] = pd.Categorical(df['Species'])
    df['Species'] = df['Species'].cat.codes

    # pre-process
    X = df.values[:, :-1]
    y = df.values[:, -1]

    # split trainSet and testSet
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # feature scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # training and prediction
    error = []
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    # evaluate the algorithm
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()


if __name__ == '__main__':
    main()


