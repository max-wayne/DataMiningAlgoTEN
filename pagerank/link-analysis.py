# -*-coding:utf-8-*-
# Ref: https://blog.csdn.net/rubinorth/article/details/52231620
#       https://www.cnblogs.com/nolonely/p/6666329.html
#       https://www.cnblogs.com/fstang/archive/2013/06/01/3113352.html

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

edge = [[1, 2], [1, 4], [1, 5], [3, 4], [4, 2], [5, 2], [5, 3]]
# edge = [[1, 2], [1, 3], [1, 4], [2, 1], [2, 4], [3, 1], [4, 2], [4, 3]]


class Pagerank(object):
    def __init__(self):
        pass

    @ staticmethod
    def calc_PR_buildin():
        G = nx.DiGraph()
        for item in edge:
            head, tail = item[0], item[1]
            G.add_edge(head, tail)

        plt.figure(1)
        layout = nx.spring_layout(G)
        nx.draw(G, pos=layout, node_color='y', with_labels=True)

        pr = nx.pagerank(G, alpha=0.85)
        for node, pagerank in pr.items():
            print('%d: %.4f' % (node, pagerank))

        plt.figure(2)
        nx.draw(G, pos=layout, node_size=[x*6000 for x in pr.values()], node_color='g', with_labels=True)
        plt.show()

    @ staticmethod
    def gen_transfer_matrix(edge):
        node = set()
        for item in edge:
            node.add(item[0])
            node.add(item[1])
        M_len = max(node)
        M = np.zeros((M_len, M_len))
        for item in edge:
            M[item[0]-1, item[1]-1] = 1
        kout = np.sum(M, axis=1)
        for ko in range(len(kout)):
            if kout[ko] != 0:
                M[ko] = M[ko] / kout[ko]
        return M.transpose()

    @ staticmethod
    def calc_PageRank(M, alpha):
        e = np.ones((M.shape[0], 1)) / M.shape[0]
        V = e
        for i in range(15):
            V = alpha * np.dot(M, V) + (1-alpha) * e

        return V / sum(V)

    def calc_PR(self):
        M = self.gen_transfer_matrix(edge)
        print(self.calc_PageRank(M, 0.85))


class HITS(object):
    def __init__(self):
        pass

    @ staticmethod
    def calc_HITS_buildin():
        G = nx.DiGraph()
        for item in edge:
            head, tail = item[0], item[1]
            G.add_edge(head, tail)

        plt.figure(1)
        layout = nx.spring_layout(G)
        nx.draw(G, pos=layout, node_color='y', with_labels=True)

        [h, a] = nx.hits(G)
        print('hub:', h)
        print('auth:', a)

        plt.figure(2)
        nx.draw(G, pos=layout, node_size=[x * 6000 for x in a.values()], node_color='g', with_labels=True)
        plt.show()

    @ staticmethod
    def gen_transfer_matrix(edge):
        node = set()
        for item in edge:
            node.add(item[0])
            node.add(item[1])
        M_len = max(node)
        M = np.zeros((M_len, M_len))
        for item in edge:
            M[item[0]-1, item[1]-1] = 1

        return M.transpose()

    @ staticmethod
    def do_HITS(M, iterations):
        A = np.ones((M.shape[0], 1)) / M.shape[0]
        H = np.ones((M.shape[0], 1)) / M.shape[0]
        for i in range(iterations):
            A = np.dot(M, H)
            H = np.dot(M.transpose(), A)
            A = A / max(A)
            H = H / max(H)

        return H/sum(H), A/sum(A)

    def calc_HITS(self):
        M = self.gen_transfer_matrix(edge)
        h, a = self.do_HITS(M, 100)
        print('hub:', h)
        print('auth:', a)


def main():
    # PR = Pagerank()
    # PR.calc_PR_buildin()
    # PR.calc_PR()
    HT = HITS()
    HT.calc_HITS()


if __name__ == '__main__':
    main()


