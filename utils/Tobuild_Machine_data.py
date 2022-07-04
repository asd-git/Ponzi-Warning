import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from process_tu_data import to_tu_file

from utils import get_files, mkdir

"生成用于机器学习的数据集"
data = 'eth'

# Todo 改这个


PUP_node = pd.read_csv('/public/zjj/jj/data/PUP_node_dataset/PUP_100up/PUP_node(400).csv')
pos = PUP_node[PUP_node['label'] == 1]['Contract'].tolist()  # 75个
neg = PUP_node[PUP_node['label'] == 0]['Contract'].tolist()

edge_nums_ls = list(range(10, 110, 10))
# 不改
root_dir = mkdir('/public/zjj/jj/data1/eth/pup/To_Machine/')
# trans_feature
trans_feature = []
for edge_nums in edge_nums_ls:
    dirname = f'/public/zjj/jj/data1/eth/pup/new_subG_dataset/{edge_nums}_edges_subG/contract_feature/'
    all_files, all_files_name = get_files(dirname, '_feature.csv')
    for account in tqdm(pos + neg):
        # print(account)
        for (file, file_name) in zip(all_files, all_files_name):
            if account == file_name.split('_')[0]:
                df = pd.read_csv(file)
                trans_feature.append(df)

ca_trans_fea = pd.concat(trans_feature,ignore_index=True)


label_ls = []
for i in range(len(edge_nums_ls)):
    ls = [1] * len(pos) + [0] * len(neg)
    label_ls.extend(ls)
ca_trans_fea['label'] = label_ls

ca_trans_fea.to_csv(root_dir + 'CA_trans_feature.csv', index=False)

# trans_feature
code_feature = []
for edge_nums in edge_nums_ls:
    code_df = pd.read_csv('/public/zjj/jj/data/PUP_node_dataset/Opcode_net_PUP.csv')

    for account in tqdm(pos + neg):
        # print(account)
        code_fea = code_df[code_df['Contract'] == account]
        code_feature.append(code_fea)

ca_code_fea = pd.concat(code_feature, ignore_index=True)

ca_code_fea.to_csv(root_dir + 'CA_code_feature.csv', index=False)
