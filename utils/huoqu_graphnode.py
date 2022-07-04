import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import get_files, mkdir

PUP_node = pd.read_csv('.././data/PUP_node_dataset/PUP_100up/PUP_node(400).csv')
pos = PUP_node[PUP_node['label'] == 1]['Contract'].tolist()  # 75个
neg = PUP_node[PUP_node['label'] == 0]['Contract'].tolist()

edge_nums_ls = list(range(10, 110, 10))
for edge_nums in edge_nums_ls:
    dirname = f'.././data_new/eth/pup/toTU/X/{edge_nums}_edges_subG'

    all_files, all_files_name = get_files(dirname, '_trans.txt')

    save_nodes_path = mkdir(f'.././data_new/eth/pup/toTU/nodes1/{edge_nums}_edges_subG')
    all_graph_node_ls = []
    for account in tqdm(pos + neg):
        print(account)
        for (file, file_name) in zip(all_files, all_files_name):
            if account == file_name.split('_')[0]:
                X_trans = np.loadtxt(file)
                nodes = X_trans.shape[0]
                all_graph_node_ls.append(nodes)


    with open(save_nodes_path+'/all_graph_node_ls.txt', 'w') as f:
        f.writelines(str(all_graph_node_ls))
        f.close()

