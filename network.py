import collections
import sys
import os

import numpy as np
import os

import networkx as nx
import matplotlib.pyplot as plt
import random

import scipy.sparse
import scipy.sparse.csgraph

from multiprocessing import Pool
import time
import itertools

dir = os.path.join(os.getcwd(), "data", "network")
prefix = "/2014_typhoon_hagupit_extended"
partial_file_path = dir + prefix
file_path = partial_file_path + "_edgelist.csv"
graph = nx.read_weighted_edgelist(file_path, create_using=nx.DiGraph())


def degreedist(graph):
    degs = graph.degree()

    degree_sequence = sorted([d for n, d in degs], reverse=False)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    outF = open(
        partial_file_path + "_degree-dist.csv",
        "w")
    outF.write("Degree\tCount")
    outF.write("\n")
    for d, c in zip(deg, cnt):
        print(d, c)
        outF.write(str(d) + "\t" + str(c))
        outF.write("\n")
    outF.close()
    print()


def nr_nodes(graph, save=False):
    users = graph.nodes()

    users = np.array(users)

    print(users[:10])

    np.random.seed(212)
    np.random.shuffle(users)

    print(users[:10])

    if save:
        outF = open(
            partial_file_path + "_nodes-" + str(len(users)) + "-shuffled-seed212.csv",
            "w")
        for user in users:
            print(user)
            outF.write(str(user))
            outF.write("\n")
        outF.close()
        print()

    print("Number of nodes: {0}".format(len(users)))
    print()


def nr_edges(graph):
    print("Number of edges: {0}".format(len(graph.edges())))
    print()


def density(graph):
    print("Density: {0}".format(nx.density(graph)))
    print()


def in_degree_dist(graph):
    indeg = graph.in_degree()
    degree_dist(indeg, "Indegree")


def users_indegree(graph):
    indegree = nx.in_degree_centrality(graph)
    indegree_sorted = sorted(indegree.items(), key=lambda x: x[1], reverse=True)
    # for node in evc_sorted:
    #     print(node[0] + ": " + str(node[1]))

    outF = open(
        partial_file_path + "_indegree.csv",
        "w")
    for user in indegree_sorted:
        print(user[0], ": ", user[1])
        outF.write(str(user[0]) + ":" + str(user[1]))
        outF.write("\n")
    outF.close()
    print()


def out_degree_dist(graph):
    indeg = graph.out_degree()
    degree_dist(indeg, "Outdegree")


def degree_dist(degs, title):
    degree_sequence = sorted([d for n, d in degs], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    plt.xscale('log')
    plt.yscale('log')
    plt.bar(deg, cnt, color='b')

    plt.title(title + " Distribution")
    plt.ylabel("Count")
    plt.xlabel(title)

    # draw graph in inset
    plt.axes([0.4, 0.4, 0.5, 0.5])
    plt.axis('off')

    plt.show()


def calculate_eigenvector(graph):
    ev_centrality = nx.eigenvector_centrality(graph)
    evc_sorted = sorted(ev_centrality.items(), key=lambda x: x[1], reverse=True)
    # for node in evc_sorted:
    #     print(node[0] + ": " + str(node[1]))

    outF = open(
        partial_file_path + "_eigenvector.csv",
        "w")
    for user in evc_sorted:
        print(user[0], ": ", user[1])
        outF.write(str(user[0]) + ":" + str(user[1]))
        outF.write("\n")
    outF.close()
    print()


def distance_distribution(graph):
    print("Distance distribution:")
    print("Size: " + str(len(graph)))
    nr_distances = {}
    for index, node in zip(range(10000), graph):
        # for index, node in enumerate(graph):
        shortest_paths = nx.single_source_shortest_path_length(graph, node)
        for to_node in shortest_paths.keys():
            dist = shortest_paths[to_node]
            if dist in nr_distances:
                nr_distances[dist] += 1
            else:
                nr_distances[dist] = 1

    plt.title("Distance Distribution")
    plt.ylabel("Count")
    plt.xlabel("Distance")
    plt.yscale('log')
    plt.bar(list(nr_distances.keys()), list(nr_distances.values()), color='b')
    plt.show()


def size_giant_component(graph):
    print("Giant component:")
    lwcc = max(nx.weakly_connected_components(graph), key=len)
    lscc = max(nx.strongly_connected_components(graph), key=len)
    print("Size largest WCC: {0} | Size largest SCC: {1}"
          .format(len(lwcc), len(lscc)))


def largest_wcc(graph):
    print("Largest WCC:")
    lwcc = max(nx.weakly_connected_components(graph), key=len)
    print("Size: " + str(len(lwcc)))


def largest_scc(graph):
    print("Largest SCC:")
    lscc = max(nx.strongly_connected_components(graph), key=len)
    print("Size: " + str(len(lscc)))


def save_centrality_file(c, file_sufix):
    outF = open(partial_file_path + file_sufix, "w")
    for user in c:
        # print(user[0],": ", user[1])
        outF.write(str(user[0]) + ":" + str(user[1]))
        outF.write("\n")
    outF.close()


def top_k_users(graph, top=20, sample_size=100000, seed=123):
    print("Betweenness centrality")
    bc_nodes = nx.betweenness_centrality(graph)
    bcn_sorted = sorted(bc_nodes.items(), key=lambda x: x[1], reverse=True)  # [:top]
    outF = open(partial_file_path + "_betweenness.csv", "w")
    for user in bcn_sorted:
        print(user[0], ": ", user[1])
        outF.write(str(user[0]) + ":" + str(user[1]))
        outF.write("\n")
    outF.close()
    print()

    print("Closeness centrality")
    # sampled_nodes = random.sample(graph.nodes(), sample_size)
    # sample_graph = nx.subgraph(graph, sampled_nodes)
    cc_nodes = nx.closeness_centrality(graph)
    ccn_sorted = sorted(cc_nodes.items(), key=lambda x: x[1], reverse=True)  # [:top]
    outF = open(partial_file_path + "_closeness.csv", "w")
    for user in ccn_sorted:
        print(user[0], ": ", user[1])
        outF.write(str(user[0]) + ":" + str(user[1]))
        outF.write("\n")
    outF.close()
    print()

    ''''print("Degree centrality")
    dc_nodes = nx.degree_centrality(graph)
    dcn_sorted = sorted(dc_nodes.items(), key=lambda x: x[1], reverse=True)[:top]
    print(dcn_sorted)
    print()'''''


def connect_nodes(first_node, param):
    pass


def split_n(n, param):
    pass


def init_graph():
    pass


def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x


def _betmap(G_normalized_weight_sources_tuple):
    """Pool for multiprocess only accepts functions with one argument.
    This function uses a tuple as its only argument. We use a named tuple for
    python 3 compatibility, and then unpack it when we send it to
    `betweenness_centrality_source`
    """
    return nx.betweenness_centrality_source(*G_normalized_weight_sources_tuple)


def betweenness_centrality_parallel(G, processes=None):
    """Parallel betweenness centrality  function"""
    p = Pool(processes=processes)
    print("processess: " + str(p._processes))
    node_divisor = len(p._pool) * 4
    node_chunks = list(chunks(G.nodes(), int(G.order() / node_divisor)))
    num_chunks = len(node_chunks)
    bt_sc = p.map(_betmap,
                  zip([G] * num_chunks,
                      [True] * num_chunks,
                      [None] * num_chunks,
                      node_chunks))

    # Reduce the partial solutions
    bt_c = bt_sc[0]
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c


def compute_betweenness_centrality():
    print("")
    print("Computing betweenness centrality for:")
    print(nx.info(graph))
    print("\tParallel version")
    start = time.time()
    bcp = betweenness_centrality_parallel(graph, 96)
    print("\t\tTime: %.4F" % (time.time() - start))
    bcp_sorted = sorted(bcp.items(), key=lambda x: x[1], reverse=True)
    save_centrality_file(bcp_sorted, "_betweenness.csv")
    # print(bcp_sorted)

    # print("\tNon-Parallel version")
    # start = time.time()
    # bc = nx.betweenness_centrality(graph)
    # print("\t\tTime: %.4F seconds" % (time.time() - start))
    # bc_sorted = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    # print(bc_sorted)
    # print("")
    #
    # nx.draw(graph)
    # plt.show()


if __name__ == "__main__":
    nr_nodes(graph)
    nr_edges(graph)
    density(graph)
    # in_degree_dist(graph)
    # out_degree_dist(graph)
    # distance_distribution(graph)
    # size_giant_component(graph)

    # top_k_users(graph)
    # calculate_eigenvector(graph)
    # users_indegree(graph)

    # degreedist(graph)
    # compute_betweenness_centrality()
