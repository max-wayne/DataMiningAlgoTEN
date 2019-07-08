# -*-coding:utf-8-*-
# @Created at: 2019-04-21 14:21
# @Author: Wayne
# Ref: https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

import random
import numpy as np
import pandas as pd


def loadDataSet(input_file, ratio):
    # read data
    df = pd.read_csv(input_file)
    df.drop('Id', axis=1, inplace=True)

    # change categorical data to number 0-2
    df['Species'] = pd.Categorical(df['Species'])
    df['Species'] = df['Species'].cat.codes

    # change DataFrame to numpy matrix
    data = df.values

    # split data into trainingSet and testSet
    trainingSet, testSet = [], []
    for i in range(len(data)):
        if random.random() < ratio:
            trainingSet.append(data[i])
        else:
            testSet.append(data[i])

    return np.array(trainingSet), np.array(testSet)


def KNN(trainingSet, testInstance, k):
    # calculate euclidean distance between testInstance and trainingSet
    testInstance = np.array(testInstance)
    distance = np.linalg.norm(trainingSet[:, 0:-1] - testInstance[0:-1], ord=2, axis=1)

    # sort distance by ascend with cluster name
    record = np.column_stack((distance, trainingSet[:, -1]))
    a_arg = np.argsort(record[:, 0])
    record = record[a_arg]

    # assign the nearest neighbor
    a = [int(e) for e in record[0:k, 1]]
    count = np.bincount(a)
    c_name = np.argmax(count)

    return c_name


def calc_precision(testSet, prediction):
    cnt = 0
    for i in range(len(testSet)):
        if testSet[i, 4] == prediction[i]:
            cnt += 1
    print(cnt/len(testSet)*100.0)


def main():
    # load data
    trainingSet, testSet = loadDataSet('../k-means/Iris.csv', 0.66)

    # generate prediction
    label_prd = []
    k = 5
    for item in testSet:
        c_n = KNN(trainingSet, item, k)
        label_prd.append(c_n)

    # calculate precision
    calc_precision(testSet, label_prd)

    # save results
    output_file = './result.csv'
    result = np.column_stack((testSet, np.array(label_prd)))
    names = ['F1', 'F2', 'F3', 'F4', 'TL', 'PL']
    result = pd.DataFrame(columns=names, data=result)
    result.to_csv(output_file, encoding='utf-8', sep='\t', index=0)


if __name__ == '__main__':
    main()






