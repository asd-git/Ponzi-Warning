#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@desc: extract ETH data from database for GNN method
# Todo: design better sample strategies
'''


import random
import numpy as np
import pickle as pkl

import pandas as pd
# from py2neo import Graph
from tqdm import tqdm
import networkx as nx
from collections import deque
import argparse
import os
from utils import *



from utils import neo4j_check_isolate, mkdir
# from utils import process_tu_data
# from process_tu_data import to_tu_file

# database config
neo4j_G = Graph("http://10.12.11.87:7475", auth=("neo4j", "000000"))

# data information
label_abbreviation = {"p": "pup"}


def Parameter():
    parser = argparse.ArgumentParser(description='Data prepare for gnn')
    parser.add_argument('-d', '--data', type=str, help='eth', default='eth')
    parser.add_argument('-l', '--label', type=str, help='p', default='p')
    # parser.add_argument('-ess', '--edge_sample_strategy', type=str, help='Volume, Times, averVolume', default='Volume')
    # parser.add_argument('--hop', type=int, default=1)
    # parser.add_argument('-k', '--topk', type=int, default=3)
    parser.add_argument('-p', '--parallel', type=int, help='parallel', default=0)
    parser.add_argument('-seed', '--seed', type=int, default=111)
    return parser.parse_args()


'''
 subgraph extract function
'''


def subgraph_extract(account, edge_nums, database, savepath):
    subgraph = nx.DiGraph()
    subgraph.add_node(account)  # treat target account as the first node

    trans_cyber = "match (s:CA{name:'" + account + "'})-[r:`trans`]->(e) " \
                                                   "return startNode(r).name as from,endNode(r).name as to, r.value as value, " \
                                                   f"r.timestamp as timestamp order by timestamp"

    call_cyber = "match (s)-[r:`call`]->(e:CA{name:'" + account + "'})" \
                                                                   "return startNode(r).name as from,endNode(r).name as to, r.value as value, " \
                                                                   f"r.timestamp as timestamp order by timestamp"
    trans_res = pd.DataFrame(database.run(trans_cyber).data())
    call_res = pd.DataFrame(database.run(call_cyber).data())
    # print(trans_res)
    # print(call_res)
    df = pd.concat([call_res, trans_res])
    # new_df = df.sort_values(['timestamp']).reset_index(drop=True)
    new_res = df.sort_values(by=['timestamp', 'from'], ascending=[True, False]).reset_index(drop=True)
    new_trans = new_res.iloc[:edge_nums, :]
    node_type = nodetype(account, database)
    if node_type == 'EOA':
        node_lifetime = '0'
    elif node_type == 'CA':
        node_lifetime = look_lifetime(account, database)
    # print("nodeType", node_type)
    # 分类合约的本身信息(特征、关系)
    sum_count_res, trans_feature_df = extract_contract_trans_info(new_trans, account, node_lifetime, savepath)
    # 获取一阶邻居的交易特征
    # contract_trans_1k_feature = extract_node_1k_trans_info(new_trans, account, node_lifetime, database, savepath)
    # 合约的代码特征 （查询即可）
    # contract_code_feature = extract_node_code_info(account)

    for i in range(sum_count_res.shape[0]):
        item = sum_count_res.iloc[i, :]
        subgraph.add_edge(item['from'], item['to'], count=float(item['count']), sum=float(item['sum']))
    # save feature
    # X_trans_df = pd.concat([trans_feature_df, contract_trans_1k_feature], ignore_index=True)
    # X_code_df = contract_code_feature
    #
    # X_trans = X_trans_df.iloc[:, 1:].to_numpy()
    # X_code = X_code_df.iloc[:, 2:].to_numpy()

    # Xs = [X_trans, X_code]

    # return subgraph, Xs, account
    return subgraph, account


def parallel_worker_of_subgraph_extract(x):
    # return subgraph_extract(*x)
    return subgraph_extract(*x)


def main():
    args = Parameter()

    label = label_abbreviation[args.label]

    # data local path
    prefix_save_path = f'.././data_new/{args.data}/'
    save_path = prefix_save_path + "{}/".format(label)
    if not os.path.exists(save_path): os.makedirs(save_path)

    if os.path.exists(save_path + '{}G_A.pkl'.format(args.data.upper())):
        print('File exist, finish!')
    else:
        print("label:{}".format(label))

    # Todo 目前只考虑大于边规模100的合约
    #  可以考虑小于100的数据文件在 /data/PUP_node_dataset/PUP_all/
    ###### file load   庞氏合约和非旁氏合约的列表
    PUP_node = pd.read_csv('.././data/PUP_node_dataset/PUP_100up/PUP_node(400).csv')
    pos = PUP_node[PUP_node['label'] == 1]['Contract'].tolist()  # 75个
    neg = PUP_node[PUP_node['label'] == 0]['Contract'].tolist()  # 325个

    #########################################################################
    #################################################################
    # subgraph extract
    Gs = []
    Xs = []
    Ys = []
    if not args.parallel:
        edge_nums_ls = list(range(10, 110, 10))
        for edge_nums in edge_nums_ls:
            all_graph_node_ls = []
            for account in tqdm(pos + neg):
                print("edge_num:{}".format(edge_nums))
                savepath = mkdir(save_path + f'/new_subG_dataset/{edge_nums}_edges_subG/')
                save_adj_path = mkdir(save_path + f'/toTU/adj/{edge_nums}_edges_subG/')
                save_X_path = mkdir(save_path + f'/toTU/X/{edge_nums}_edges_subG/')
                # sg, X, target = subgraph_extract(account, edge_nums, neo4j_G, savepath)
                sg, target = subgraph_extract(account, edge_nums, neo4j_G, savepath)
                Gs.append(sg)
                # Xs.append(X)

                # all_graph_node_ls.append(X[0].shape[0])
                Ys += [1] if account in pos else [0]
                nx.write_gml(sg, save_adj_path + f"{target}_graph.gml")
                # np.savetxt(save_X_path + f'{target}_X_trans.txt', X[0])
                # np.savetxt(save_X_path + f'{target}_X_code.txt', X[1])
            save_nodes_path = mkdir(save_path + f'/toTU/nodes/{edge_nums}_edges_subG/')
            # Todo 这里的保存有点问题，可以看  huoqu_graphnode.py
            with open(save_nodes_path + '/all_graph_node_ls.txt', 'w') as f:
                f.writelines(str(all_graph_node_ls))
                f.close()
            ##########################################################
            save_Y_path = mkdir(save_path + f'/toTU/Y/{edge_nums}_edges_subG/')
            with open(save_Y_path + '/Y.txt', 'w') as f1:
                f1.writelines(str(Ys))
                f1.close()
    # to_tu_file(G_list=Gs, X_list=Xs, gy_list=Ys, path=save_path, edges_nums_ls=edge_nums_ls,
    #            all_graph_node_ls=all_graph_node_ls, dataname=args.data, target_node2label=account2label)



if __name__ == '__main__':
    # todo # 因为数据比较大，所以保存下来再生成图，具体见  Tobuild_Tu_data_concat.py

    main()
