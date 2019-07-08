# -*-coding:utf-8-*-
# @Created at: 2019-04-20 11:22
# @Author: Wayne
# Ref: 1. https://www.kaggle.com/andyxie/k-means-clustering-implementation-in-python
#      2. https://mubaris.com/posts/kmeans-clustering/
#      3. https://www.jianshu.com/p/3bb2cc453df1

from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def k_means(data, k, cate):
    """
    This function implement k-means based on calculate euclidean distance.
    Parameters
    ----------
    data: data need to be clustered
    k: number of cluster
    cate: label

    Returns
    ----------
    None

    Notes
    ----------
    If features too much, it's difficult to recognize each cluster intuitively in 2-D or 3-D.
    But we can use FactorAnalysis or PCA to reduce dimension
    """
    n = data.shape[0]   # number of training data
    c = data.shape[1]   # number of features

    # generate random centers, here use sigma and mean to ensure it represent the whole data?
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    centers = np.random.randn(k, c) * std + mean

    # plot data
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = ['orange', 'green', 'blue']
    for i in range(n):
        ax.scatter(data[i, 0], data[i, 1], data[i, 2], s=10, color=colors[int(cate[i])])

    centers_old = np.zeros(centers.shape)   # to store old centers
    centers_new = deepcopy(centers)

    distances = np.zeros((n, k))

    error = np.linalg.norm(centers_new - centers_old)

    # when the estimate of that center stays the same, exit loop
    while error != 0:
        # measure the distance to each center
        for i in range(k):
            # euclidean distance
            distances[:, i] = np.linalg.norm(data - centers_new[i], axis=1)
        # assign all training data to closest center
        clusters = np.argmin(distances, axis=1)

        centers_old = deepcopy(centers_new)
        # calculate mean for every cluster and update the center
        for i in range(k):
            centers_new[i] = np.mean(data[clusters == i], axis=0)
        error = np.linalg.norm(centers_new - centers_old)
    print(centers_new)
    ax.scatter(centers_new[:, 0], centers_new[:, 1], centers_new[:, 2], marker='*', c='r', s=100)
    plt.show()


def generate_random_data():
    # set three centers
    center_1 = np.array([1, 1])
    center_2 = np.array([5, 5])
    center_3 = np.array([8, 1])

    # generate random data and center it to the three centers
    data_1 = np.random.randn(200, 2) + center_1
    data_2 = np.random.randn(200, 2) + center_2
    data_3 = np.random.randn(200, 2) + center_3

    data = np.concatenate((data_1, data_2, data_3), axis=0)
    k_means(data, 3)


def iris_demo():
    # load data
    df = pd.read_csv('./Iris.csv')
    df.drop('Id', axis=1, inplace=True)

    # change categorical data to number 0-2
    df['Species'] = pd.Categorical(df['Species'])
    df['Species'] = df['Species'].cat.codes

    # change DataFrame to numpy matrix
    data = df.values[:, 0:4]
    category = df.values[:, 4]

    k_means(data, 3, category)

    kmeans = KMeans(n_clusters=3)
    kmeans = kmeans.fit(data)
    labels = kmeans.predict(data)
    centroid = kmeans.cluster_centers_

    print(centroid)


def main():
    iris_demo()


if __name__ == '__main__':
    main()

