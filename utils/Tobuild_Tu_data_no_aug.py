import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from process_tu_data import to_tu_file

from utils import get_files, mkdir

data = 'eth'

# Todo 改这个
prefix_save_path = f'./data_no_Aug/{data}/'
save_path = prefix_save_path
PUP_node = pd.read_csv('.././data/PUP_node_dataset/PUP_100up/PUP_node(400).csv')
pos = PUP_node[PUP_node['label'] == 1]['Contract'].tolist()  # 75个
neg = PUP_node[PUP_node['label'] == 0]['Contract'].tolist()

edge_nums = 100
# 不改
root_dir = '.././data_new/eth/pup/toTU'
# Gs
Gs = []

dirname = root_dir + f'/adj/{edge_nums}_edges_subG'
all_files, all_files_name = get_files(dirname, '_graph.gml')
for account in tqdm(pos + neg):
    # print(account)
    for (file, file_name) in zip(all_files, all_files_name):
        if account == file_name.split('_')[0]:
            graphG = nx.read_gml(file)
            Gs.append(graphG)

# Xs
Xs = []

dirname = root_dir + f'/X/{edge_nums}_edges_subG'
all_trans_files, all_trans_files_name = get_files(dirname, '_trans.txt')
all_code_files, all_code_files_name = get_files(dirname, '_code.txt')
for account in tqdm(pos + neg):
    for (file, file_name, code_file, code_file_name) in zip(all_trans_files, all_trans_files_name, all_code_files,
                                                            all_code_files_name):
        if account == file_name.split('_')[0]:
            X_trans = np.loadtxt(file)
            X_code = np.loadtxt(code_file)
            Xs.append([X_trans, X_code])

# all_graph_node_ls
all_graph_node_ls = []

with open(root_dir + f"/nodes/{edge_nums}_edges_subG/"
                     "all_graph_node_ls.txt", 'r') as f:
    a = f.read()
    # print(a)
    res = a.strip('[')
    res = res.strip(']')
    res = res.split(',')
    node_ls = [int(r) for r in res]
    all_graph_node_ls.extend(node_ls)
f.close()

# Ys
Ys = None
with open(root_dir + f"/Y/{edge_nums}_edges_subG/Y.txt", 'r') as f:
    a = f.read()
    print(a)
    res = a.strip('[')
    res = res.strip(']')
    res = res.split(',')
    Y_ls = [int(r) for r in res]
    Ys = Y_ls[:400]

account2label = list(zip(pos + neg, Ys))
to_tu_file(G_list=Gs, X_list=Xs, gy_list=Ys, path=save_path, edges_nums_ls=None,
           all_graph_node_ls=all_graph_node_ls, dataname=data, target_node2label=account2label)
