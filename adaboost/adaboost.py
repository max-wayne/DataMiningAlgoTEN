# -*-coding:utf-8-*-
# @Created at: 2019-05-04 15:28
# @Author: Wayne
# Ref: https://zhuanlan.zhihu.com/p/41536315
#       https://blog.csdn.net/v_july_v/article/details/40718799

# -----------------------------------------------------------------
# Steps:
# 1.初始化训练数据的权值分布，1/N.
# 2.训练弱分类器。如果某个样本点被准确分类，那么在构造下一个训练集中，它的
#   权值将被降低；相反，如果被错误分类，它的权值将被提高。更新后的权值样本
#   被用于下一个分类器，整个训练过程迭代进行下去。
# 3.将各个弱分类器组合成强分类器。各个分类器的训练过程结束后，加大分类误差
#   率小的弱分类器的权重；降低分类误差率大的弱分类器的权重。
# Equations:
# 初始权重: D_1 = (w_11, w_12, ..., w_1N)
# 分类器: G_m(x): X-->{-1, +1}
# 分类误差率: e_m = P(G_m(xi)!=yi) = Sigma_i=1,N(w_mi)I(G_m(xi)!=yi)
# 分类器权重: alpha_m = 1/2*ln((1-e_m)/e_m)
# 更新样本权值分布: D_m+1 = (w_(m+1, 1), w_(m+1, 2), ..., w_(m+1, N))
#                w_(m+1, i) = w_(m,i)/Z_m*exp(-alpha*yi*G_m(xi))
#                Z_m = Sigma(w_(m,i)*exp(-alpha*yi*G_m(xi)))
# 组合弱分类器: f(x) = Sigma_(m=1, N)(alpha_m*G_m(x))
# 最终分类器: G(x) = sign(f(x))
# -----------------------------------------------------------------

import math
import numpy as np


def sign(x, v):
    """
    Signal function.
    Args:
         x: data.
         v: threshold.
    Returns:
        1 if x < threshold.
        0 if x > threshold.
    """
    if x < v:
        return 1
    if x > v:
        return -1


def loadData():
    """
    load data set.
    """
    x = [i for i in range(10)]
    y = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]

    return x, y


def find_weak_clf(X, Y, D_i, cand_v):
    """
    Find the best threshold which makes the error rate lowest.
    Args:
        X: dataSet.
        Y: labels.
        D_i: sample weights of the i-th iteration.
        cand_v: candidate threshold.
    Returns:
        Best threshold v.
    """
    e_m = 1.0  # 初始化分类误差率
    best_v = 1.0  # 初始化分类阈值
    sign_f = 0  # 初始化符号函数（是否使用原函数）,0：原函数 1：条件对调
    Y_pred = [0] * len(X)  # 初始化预测值

    # find best threshold v
    for v in cand_v:
        flag = 0
        y_pred = [sign(x, v) for x in X]
        res = np.array(y_pred) + np.array(Y)
        # 分类误差率等于分错样本权重之和
        e = sum([D_i[i] for i in range(len(res)) if res[i] == 0])
        if e == 0.5:  # 与随机预测相同
            continue
        if e > 0.5:   # 将符号函数判别条件对调，flag置1.
            e = 1 - e
            flag = 1
        if e < e_m:
            e_m = round(e, 4)
            best_v = v
            Y_pred = y_pred
            sign_f = flag

    # update D_i
    alpha_m = round(math.log((1-e_m)/e_m) / 2, 4)
    Z_m = round(sum([D_i[i]*math.exp(-alpha_m*Y[i]*Y_pred[i]) for i in range(len(X))]), 4)
    D_i_new = [round(D_i[i]/Z_m*math.exp(-alpha_m*Y[i]*Y_pred[i]), 4) for i in range(len(X))]

    return best_v, sign_f, e_m, alpha_m, D_i_new


def is_stop(X, Y, M):
    """
    Decide whether stop or not based on G(x)'s performance.
    Args:
        X: dataSet.
        Y: labels.
        M: present model.
    Returns:
        The number of samples that be predicted error.
    """
    Y_pred = list()
    for i in range(len(X)):
        y_temp = 0
        for j in range(len(M)):
            v, sign_f, alpha = M[j][0], M[j][1], M[j][2]
            if sign_f == 0:
                y_temp += alpha * sign(X[i], v)
            else:
                y_temp += -alpha * sign(X[i], v)
        if y_temp < 0:
            Y_pred.append(-1)
        else:
            Y_pred.append(1)
    cnt = 0
    for i in range(len(Y)):
        if Y[i] != Y_pred[i]:
            cnt += 1

    return cnt


def train_classifier(X, Y):
    """
    Generate strong classifier.
    Args:
        X: dataSet.
        Y: labels
    Return:
        A strong classifier.
    """
    # initialize
    D_1 = [1 / len(X)] * len(X)
    cand_v = np.arange(min(X), max(X), 0.5)
    cand_v = [v for v in cand_v if v*2 % 2 == 1]

    # calculate
    D, M = list(), list()  # save sample weights and model for each iteration
    D.append(D_1)
    G_x = 1.0
    i = 0
    while G_x != 0:  # 样本在G(x)下是否还有分错的点
        i += 1
        v, sign_f, error, alpha, D_i = find_weak_clf(X, Y, D[i-1], cand_v)
        D.append(D_i)
        M.append([v, sign_f, alpha, error])
        G_x = is_stop(X, Y, M)

    return D, M


def main():
    # load data
    X, Y = loadData()

    # training
    D, M = train_classifier(X, Y)

    # print weights of dataSet
    for item in D:
        print(item)

    # print model parameter
    for item in M:
        print('threshold={0}, sign_f={1}, alpha={2}, error={3}'.format(item[0], item[1],
                                                                       item[2], item[3]))


if __name__ == '__main__':
    main()


