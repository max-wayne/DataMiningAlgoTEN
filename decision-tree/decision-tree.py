# -*-coding:utf-8-*-
# @Created at: 2019-04-24 22:57
# @Author: Wayne
# Ref: http://www.cnblogs.com/wsine/p/5180315.html

import math
from collections import Counter


def calcShannonEnt(dataSet):
    """
    input: 数据集
    output: 数据集的熵
    notes: 计算给定数据集的熵，熵越大，混乱程度越大
    """
    labels = [item[-1] for item in dataSet]
    labelsCnt = Counter(labels)
    p_labels = [labelsCnt[e]/len(labels) for e in labelsCnt]
    shannonEnt = sum([-e*math.log(e, 2) for e in p_labels])
    return shannonEnt


def calcGini(dataSet):
    """
    input: 数据集
    output: 数据集的Gini值
    notes: 计算数据集的Gini, 值越小，样本纯度越高（越不容易被分错）
    """
    labels = [item[-1] for item in dataSet]
    labelsCnt = Counter(labels)
    p_labels = [labelsCnt[e]/len(labels) for e in labelsCnt]

    return 1-sum([math.pow(p, 2) for p in p_labels])


def splitDataSet(dataSet, axis, value):
    """
    input: 数据集，特征维度，特征值
    output: 划分数据集
    notes: 返回等于选择特征值的记录
    """
    subDataSet = [item[:axis]+item[axis+1:] for item in dataSet if item[axis] == value]
    return subDataSet


def chooseBestFeatureToSplit_C45(dataSet):
    """
    input: 数据集
    output: 最好划分特征
    notes: 基于信息增益率，选择最好的数据集划分维度
    """
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGainRatio = 0.0
    bestFeature = -1
    # check every feature
    for i in range(numFeatures):
        featureList = [item[i] for item in dataSet]
        uniqueVal = set(featureList)    # 该特征下的所有特征值
        newEntropy = 0.0
        splitInfo = 0.0
        for v in uniqueVal:
            subDataSet = splitDataSet(dataSet, i, v)
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
            splitInfo += -prob * math.log(prob, 2)
        infoGain = baseEntropy - newEntropy
        if splitInfo == 0:  # 该特征对划分无贡献
            continue
        infoGainRatio = infoGain / splitInfo
        if infoGainRatio > bestInfoGainRatio:
            bestInfoGainRatio = infoGainRatio
            bestFeature = i
    return bestFeature


def chooseBestFeatureToSplit_Gini(dataSet):
    """
    input: 数据集
    output: 最好划分特征
    notes: 基于Gini值，选择最好的数据集划分维度
            实现存在一定问题，因为CART构建的树是二叉树，这里明显不符合，后续更改。。。
    """
    numFeatures = len(dataSet[0]) - 1
    bestGini = 1e4
    bestFeature = -1
    # check every feature
    for i in range(numFeatures):
        featureList = [item[i] for item in dataSet]
        uniqueVal = set(featureList)
        newGini = 0.0
        for v in uniqueVal:
            subDataSet = splitDataSet(dataSet, i, v)
            prob = len(subDataSet) / len(dataSet)
            newGini += prob * calcGini(subDataSet)
        if newGini < bestGini:
            bestGini = newGini
            bestFeature = i

    return bestFeature


def majorityCnt(labelsList):
    """
    input: 标签列表
    output: 子节点的分类
    notes: 数据集已经处理了所有的属性，但是类标签依然不是唯一的，
            则采用多数判决的方法决定该子节点的分类
    """
    count = Counter(labelsList)
    return count.most_common(1)[0][0]


def createTree(dataSet, labels):
    """
    input: dataSet, labels
    output: decision tree
    """
    labelsList = [item[-1] for item in dataSet]
    if labelsList.count(labelsList[0]) == len(labelsList):
        # 若数据集的所有标签都一样
        return labelsList[0]
    if len(dataSet[0]) == 1:
        # 遍历完所有特征时,返回出现次数最多的label
        return majorityCnt(labelsList)
    bestFeature = chooseBestFeatureToSplit_C45(dataSet)
    # bestFeature = chooseBestFeatureToSplit_Gini(dataSet)
    bestFeatureLabel = labels[bestFeature]
    decisionTree = {bestFeatureLabel: {}}
    del(labels[bestFeature])

    featureValues = [item[bestFeature] for item in dataSet]
    uniqueVal = set(featureValues)
    for v in uniqueVal:
        subLabels = labels[:]
        decisionTree[bestFeatureLabel][v] = createTree(splitDataSet(dataSet, bestFeature, v), subLabels)

    return decisionTree


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


def classify(inputTree, featLabels, testVec):
    """
    input: 决策树，分类标签，测试数据
    output: 分类结果
    """
    firstStr = list(inputTree.keys())[0]    # 决策树的第一个特征
    secondDict = inputTree[firstStr]    # 以第一个特征划分后的决策树
    featIndex = featLabels.index(firstStr)  # 特征的索引值
    classLabel = None
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':    # 非叶子节点，继续划分
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]

    return classLabel


def classifyAll(inputTree, featLabels, testDataSet):
    """
    input: 决策树，分类标签，测试数据
    output: 分类结果
    """
    classLabelsAll = []
    for testVec in testDataSet:
        classLabelsAll.append(classify(inputTree, featLabels, testVec))

    return classLabelsAll


def createDataSet():
    """
    outlook-> 0:sunny | 1:overcast | 2: rain
    temperature-> 0:hot | 1:mid | 2:cool
    humidity-> 0:high | 1:normal
    windy-> 0:false | 1:true
    """
    dataSet = [
        [0, 0, 0, 0, 'N'],
        [0, 0, 0, 1, 'N'],
        [1, 0, 0, 0, 'Y'],
        [2, 1, 0, 0, 'Y'],
        [2, 2, 1, 0, 'Y'],
        [2, 2, 1, 1, 'N'],
        [1, 2, 1, 1, 'Y']]
    labels = ['outlook', 'temperature', 'humidity', 'windy']
    return dataSet, labels


def createTestSet():
    """
    outlook-> 0:sunny | 1:overcast | 2: rain
    temperature-> 0:hot | 1:mid | 2:cool
    humidity-> 0:high | 1:normal
    windy-> 0:false | 1:true
    """
    testSet = [
        [0, 1, 0, 0],
        [0, 2, 1, 0],
        [2, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 1, 0, 1],
        [1, 0, 1, 0],
        [2, 1, 0, 1]]
    return testSet


def main():
    dataSet, labels = createDataSet()
    labels_tmp = labels[:]
    decisionTree = createTree(dataSet, labels_tmp)
    print(decisionTree)
    storeTree(decisionTree, './tree-C45')
    grabTree('./tree-C45')
    testSet = createTestSet()
    print('classifyResult:\n', classifyAll(decisionTree, labels, testSet))


if __name__ == '__main__':
    main()




