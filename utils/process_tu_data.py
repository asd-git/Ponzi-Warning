#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: process_tu_data.py
@time: 2021/12/28 9:55
@desc:
'''
import numpy as np
import json
import pandas as pd
import networkx as nx
from scipy.io import loadmat
import scipy.sparse as sp
from itertools import combinations
from tqdm import tqdm
import random
# import sn
import os
import os.path as osp
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.io import read_planetoid_data
from torch_geometric.datasets import TUDataset


# from gensim.models.doc2vec import TaggedDocument
# from gensim.models import Doc2Vec


def to_tu_file(G_list: list, X_list: list, gy_list: list, path: str,
               edges_nums_ls=None,
               all_graph_node_ls=None,
               dataname=None,
               target_node2label=None):
    '''
    save link-subgraphs to tu_file
    :param G_list:
    :param X_list:
    :param gy_list:
    :param DS:
    :param ny_list:
    :return:
    '''
    save_path = '{}/raw/'.format(path)
    if not os.path.exists(save_path): os.makedirs(save_path)
    DS = dataname.lower()

    A1_list = [nx.adj_matrix(g, weight="count") for g in G_list]
    A2_list = [nx.adj_matrix(g, weight="sum") for g in G_list]
    A1_block = sp.block_diag(A1_list)
    A2_block = sp.block_diag(A2_list)
    rows, cols, e_attr_1 = sp.find(A1_block)
    _, _, e_attr_2 = sp.find(A2_block)

    # edge list
    print('write {}_A.txt and {}_edge_attributes.txt file...'.format(DS, DS))
    with open(save_path + '{}_A.txt'.format(DS), 'w') as f1:
        with open(save_path + '{}_edge_attributes.txt'.format(DS), 'w') as f2:
            for u, v, ea1, ea2 in zip(rows, cols, e_attr_1, e_attr_2):
                f1.writelines('{}, {}\n'.format(u + 1, v + 1))
                f2.writelines('{}, {}\n'.format(ea1, ea2))
    f1.close()
    f2.close()

    # graph_indicator.txt

    graph_indicators = []
    for i, node_len in enumerate(all_graph_node_ls):
        graph_indicator = np.ones(shape=(1, node_len)) * (i + 1)
        graph = graph_indicator.tolist()[0]
        graph_indicators.extend(graph)
    print(len(graph_indicators))

    # print(graph_indicators)
    with open(save_path + '{}_graph_indicator.txt'.format(DS), 'w') as f:
        for ind in graph_indicators:
            f.writelines("{}\n".format(int(ind)))

    # graph labels
    print('write {}_graph_labels.txt file...'.format(DS))
    with open(save_path + '{}_graph_labels.txt'.format(DS), 'w') as f:
        for y in gy_list:
            f.writelines('{}\n'.format(y))
    f.close()

    # # graph_sizes.txt
    # print('write {}_graph_sizes.txt file...'.format(DS))
    # if edges_nums_ls:
    #     graph_sizes = []
    #     for edges_nums in edges_nums_ls:
    #         for i in range(0, 400):
    #             graph_sizes.append(edges_nums)
    #
    #     with open(save_path + '{}_graph_sizes.txt'.format(DS), 'w') as f:
    #         for size in graph_sizes:
    #             f.writelines(str(size) + " " + "\n")
    #     f.close()



    # node feature
    print('write {}_node_attributes.txt file...'.format(DS))

    X_trans = np.vstack([X_list[i][0]for i in range(len(X_list))])
    np.savetxt(save_path + '{}_node_attributes.txt'.format(DS), X=X_trans, fmt='%f', delimiter=',')



    # code feature
    print('write {}_code_attributes.txt file...'.format(DS))
    X_code = np.vstack([X_list[i][1]for i in range(len(X_list))])
    np.savetxt(save_path + '{}_code_attributes.txt'.format(DS), X=X_code, fmt='%f', delimiter=',')

    # all feature
    print('write {}_all_attributes.txt file...'.format(DS))

    X_all = np.vstack([X_list[i][2] for i in range(len(X_list))])
    np.savetxt(save_path + '{}_node_attributes.txt'.format(DS), X=X_all, fmt = '%f', delimiter = ',')

    # print('write {}_target_edge.txt file...'.format(DS))
    # if target_node2label:
    #     with open(save_path + '{}_target_node2label.txt'.format(DS), 'w') as f:
    #         for node2label in target_node2label:
    #             f.writelines('{} {}\n'.format(node2label[0], node2label[1]))
    #     f.close()
