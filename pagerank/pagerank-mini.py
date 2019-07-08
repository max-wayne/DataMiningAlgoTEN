# -*-coding:utf-8-*-
# @Created at: 2019-05-03 17:27
# @Author: Wayne
# Ref: https://www.cnblogs.com/fstang/archive/2013/06/01/3113352.html


import networkx as nx
from matplotlib import pyplot as plt


G = nx.DiGraph()

with open('./test.csv', 'r') as read_f:
    for line in read_f:
        head, tail = [str(x) for x in line.strip().split('\t')]
        G.add_edge(head, tail)

layout = nx.spring_layout(G)
plt.figure(1)
nx.draw(G, pos=layout, node_colot='y', with_labels=True)

pr = nx.pagerank(G, alpha=0.85)

for node, value in pr.items():
    print(node, value)

plt.figure(2)
nx.draw(G, pos=layout, node_size=[x * 6000 for x in pr.values()], node_color='g', with_labels=True)
plt.show()


