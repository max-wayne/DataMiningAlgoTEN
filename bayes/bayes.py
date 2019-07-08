# -*-coding:utf-8-*-
# @Created at: 2019-05-03 10:55
# @Author: Wayne
# Ref: https://github.com/max-wayne/AiLearning/blob/dev/blog/ml/4.朴素贝叶斯.md

import math
import numpy as np


def loadDataSet():
    """
    Create DataSet.
    Returns:
        List of posting List and class Vec.
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]   # 1 is abusive, 0 otherwise.

    return postingList, classVec


def createVocabList(dataSet):
    """
    Create the set of all words.
    Args:
        dataSet: records of comments.
    Returns:
        Set of all words.
    """
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)

    return list(vocabSet)


def setWords2Vec(vocabList, inputSet):
    """
    Check each word in inputSet whether present or not in vocabList.
    Args:
        vocabList: set of all words.
        inputSet: input data.
    Returns:
        A vector of each word in inputSet whether present or not in vocabList.
        [0, 1, ..., 1] 1: present; 0: otherwise.
    """
    resultVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            resultVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my Vocabulary!' % word)

    return resultVec


def trainNB(trainMatrix, trainCategory):
    """
    Training from training dataSet.
    Args:
        trainMatrix: matrix of document.
        trainCategory: class of document.
    Returns:
        p0Vec, p1Vec: the probability of each word belongs to normal or abusive.
        pAbusive: the ratio of abusive document.
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])

    # the probability of abusive document
    pAbusive = sum(trainCategory) / numTrainDocs

    # p0Num: normal, p1Num: abusive
    p0Num, p1Num = np.ones(numWords), np.ones(numWords)

    # p0NumAll: normal, p1NUmAll: abusive
    p0NumAll, p1NumAll = 2, 3
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1NumAll += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0NumAll += sum(trainMatrix[i])

    # class 1: abusive, class 0: normal
    p1Vec = np.log(p1Num/p1NumAll)
    p0Vec = np.log(p0Num/p0NumAll)

    return p0Vec, p1Vec, pAbusive


def classNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    Given a test document, return the class it most likely belongs to.
    Args:
        vec2Classify: a test document.
        p0Vec: the probability of each word belongs to the Class of normal.
        p1Vec: the probability of each word belongs to the Class of abusive.
        pClass1: the probability of abusive document.
    Returns:
        Class 0 or 1.
    """
    # calc p(ci|w) = p(w|ci) * p(ci) / p(w)
    #              = p(w1|ci)*p(w2|ci)*...*p(wn|ci) * p(ci) / p(w)
    #              = sigma_{i=1}^{n}(log(p(wi|ci))) * p(ci) / p(w)
    p1 = sum(vec2Classify * p1Vec) + math.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + math.log(1-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def main():
    # load data
    listPost, listClass = loadDataSet()

    # create words set
    myVocabList = createVocabList(listPost)

    # create vector of document in 0/1 format.
    trainMat = []
    for doc in listPost:
        trainMat.append(setWords2Vec(myVocabList, doc))

    # training.
    p0V, p1V, pAb = trainNB(np.array(trainMat), np.array(listClass))

    # testing.
    s1 = ['love', 'my', 'dalmation']
    t1 = np.array(setWords2Vec(myVocabList, s1))
    print(s1, 'classified as: ', classNB(t1, p0V, p1V, pAb))

    s2 = ['stupid', 'garbage']
    t2 = np.array(setWords2Vec(myVocabList, s2))
    print(s2, 'classified as: ', classNB(t2, p0V, p1V, pAb))


if __name__ == '__main__':
    main()

