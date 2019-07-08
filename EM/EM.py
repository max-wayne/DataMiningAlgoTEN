# -*-coding:utf-8-*-
# @Created at: 2019-04-15 21:56
# @Author: Wayne
# A demo of Expectation Maximization Algorithm
# Ref: <<What is the expectation maximization algorithm?>>

import numpy as np
import math


def EM(R, p_a, p_b):
    row, col = len(R), len(R[0])
    # A_or_B记录每一轮实验选中A硬币还是B硬币的概率
    A_or_B = np.zeros((row, 2))
    for i in range(row):
        a = math.pow(p_a, sum(R[i][:]))*math.pow(1-p_a, col-sum(R[i][:]))
        b = math.pow(p_b, sum(R[i][:]))*math.pow(1-p_b, col-sum(R[i][:]))
        A_or_B[i][0] = round(a/(a+b), 3)
        A_or_B[i][1] = round(b/(a+b), 3)
    # R_A(R_B)记录每一轮实验选中A(B)硬币时正面、反面出现的次数
    R_A = np.zeros((row, 2))
    R_B = np.zeros((row, 2))
    for i in range(row):
        R_A[i][0] = A_or_B[i][0] * sum(R[i][:])
        R_A[i][1] = A_or_B[i][0] * (col-sum(R[i][:]))
        R_B[i][0] = A_or_B[i][1] * sum(R[i][:])
        R_B[i][1] = A_or_B[i][1] * (col-sum(R[i][:]))
    # 更新硬币A(B)正面朝上的概率
    p_a = round(sum(R_A)[0]/sum(sum(R_A)), 3)
    p_b = round(sum(R_B)[0]/sum(sum(R_B)), 3)
    return p_a, p_b


def main():
    # rec记录每一轮实验硬币情况, head=1, tail=0
    rec1 = [1, 0, 0, 0, 1, 1, 0, 1, 0, 1]
    rec2 = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
    rec3 = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1]
    rec4 = [1, 0, 1, 0, 0, 0, 1, 1, 0, 0]
    rec5 = [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]
    rec = np.array([rec1, rec2, rec3, rec4, rec5])
    # 初始化硬币正面朝上的概率
    theta_A, theta_B = 0.6, 0.5
    epsilon = 1e-3
    delta = 1
    while delta > epsilon:
        a, b = EM(rec, theta_A, theta_B)
        delta = abs(theta_A-a) + abs(theta_B-b)
        theta_A, theta_B = a, b
    print(theta_A, theta_B)


if __name__ == '__main__':
    main()
