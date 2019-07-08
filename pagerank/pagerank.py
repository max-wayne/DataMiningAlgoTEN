# -*-coding:utf-8-*-
# @Created at: 2019-05-03 13:36
# @Author: Wayne
# Ref: https://liam.page/2017/04/04/Python-100-lines-of-PageRank/(原文有错误)
#       https://blog.csdn.net/rubinorth/article/details/52215036


class Vertex:
    def __init__(self):
        self.in_degree = 0
        self.out_degree = 0
        self.pagerank = 0.0


class Edge:
    def __init__(self, start, end):
        self.start_id = start
        self.end_id = end


def addVertex(vertexName, vtx_map):
    """
    Add vertex.
    Args:
        vertexName: url of a web page.
        vtx_map: the map from url to id.
    Returns:
        the id of vertex.
    """
    if vertexName in vtx_map:
        return vtx_map[vertexName]
    else:
        res_id = len(vtx_map)
        vtx_map[vertexName] = res_id

    return res_id


def readTable(f_name, vtx_map, edge_list):
    """
    read f_name line by line, update vtx_map and edge_list.
    Args:
        f_name: input file name.
        vtx_map: the map from url to id.
        edge_list: the list of all edges.
    Returns:
        None.
    """
    with open(f_name, 'r') as read_f:
        for line in read_f:
            s = line.strip().split('\t')
            start = addVertex(s[0], vtx_map)
            end = addVertex(s[1], vtx_map)
            edge_list.append(Edge(start, end))

    return None


def initialize(vtx_map, edge_list):
    """
    Initialize the data structure.
    Args:
        vtx_map: the map from url to id.
        edge_list: the list of all edges.
    Returns:
        vtx_list.
    """
    vtx_num = len(vtx_map)
    assert vtx_num > 0
    vtx_list = [Vertex() for _ in range(vtx_num)]
    for i in range(vtx_num):
        vtx_list[i].pagerank = 1.0 / vtx_num
    for edge in edge_list:
        vtx_list[edge.start_id].out_degree += 1
        vtx_list[edge.end_id].in_degree += 1

    return vtx_list


def calcPageRank(alpha, num_iter, vtx_map, edge_list):
    """
    Calc each node's PageRank.
    Args:
        alpha: the probability of searching a page from present.
        num_iter: the times of calculation iterative.
        vtx_map: the map from url to id.
        edge_list: the list of all edges.
    Returns:

    """
    vtx_list = initialize(vtx_map, edge_list)
    vtx_num = len(vtx_list)
    assert vtx_num > 0
    pr_list = [item.pagerank for item in vtx_list]
    damping_value = (1 - alpha) / vtx_num
    for _ in range(num_iter):
        # calc
        for edge in edge_list:
            pr_list[edge.end_id] += alpha * vtx_list[edge.start_id].pagerank / \
                                         vtx_list[edge.start_id].out_degree
        for edge in edge_list:
            pr_list[edge.end_id] += alpha * vtx_list[edge.start_id].pagerank / \
                                    vtx_list[edge.start_id].out_degree
        # update
        for i in range(vtx_num):
            vtx_list[i].pagerank = (pr_list[i] + damping_value) / sum(pr_list)

    return vtx_list


def startPageRank(f_name, alpha=0.85, num_iter=20):
    vtx_map = dict()
    edge_list = list()
    readTable(f_name, vtx_map, edge_list)
    Vtx = calcPageRank(alpha, num_iter, vtx_map, edge_list)
    print('node PageRank')
    for key in vtx_map:
        print(key, Vtx[vtx_map[key]].pagerank)


def main():
    startPageRank('./test.csv')


if __name__ == '__main__':
    main()

