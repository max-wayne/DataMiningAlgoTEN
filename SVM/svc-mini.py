# -*-coding:utf-8-*-
# @Created at: 2019-04-30 22:11
# @Author: Wayne
# Ref: https://juejin.im/post/5b7fd39af265da43831fa136

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def load_data():
    bank_data = pd.read_csv('./data_banknote_authentication.txt', sep=',')
    X = bank_data.drop('Class', axis=1)
    y = bank_data['Class']

    return X, y


def main():
    # load data
    X, y = load_data()

    # split train_set, test_set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # training model
    svc_classifier = SVC(kernel='linear')
    svc_classifier.fit(X_train, y_train)

    # predict
    y_pred = svc_classifier.predict(X_test)

    # evaluate algorithm
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()



