import os
import copy

import pandas as pd
import torch
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from os.path import join as pjoin


# from parser import args

def split_ids(ids, folds=10):
    if folds > 0:
        n = len(ids)
        stride = int(np.ceil(n / float(folds)))
        test_ids = [ids[i: i + stride] for i in range(0, n, stride)]
        assert np.all(
            np.unique(np.concatenate(test_ids)) == sorted(ids)), 'some graphs are missing in the test sets'
        assert len(test_ids) == folds, 'invalid test sets'
        train_ids = []
        for fold in range(folds):
            train_ids.append(np.array([e for e in ids if e not in test_ids[fold]]))
            assert len(train_ids[fold]) + len(test_ids[fold]) == len(
                np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == n, 'invalid splits'
    else:
        folds = -folds
        n = len(ids)
        stride = int(np.ceil(n / float(folds)))
        train_ids = [ids[i: i + stride] for i in range(0, n, stride)]
        assert np.all(
            np.unique(np.concatenate(train_ids)) == sorted(ids)), 'some graphs are missing in the train sets'
        assert len(train_ids) == folds, 'invalid train sets'
        test_ids = []
        for fold in range(folds):
            test_ids.append(np.array([e for e in ids if e not in train_ids[fold]]))
            assert len(train_ids[fold]) + len(test_ids[fold]) == len(
                np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == n, 'invalid splits'
    return train_ids, test_ids


# 特征矩阵归一化
def normalize_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    # features = sp.coo_matrix(features)
    rowsum = np.array(features.max(1) + 1e-5, dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


# 列归一化
def normalization_col(X):
    col_sum = np.sum(X, axis=0)
    return X / col_sum


def mkdir(path):
    """
    :param path:
    :return:
    """
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    return path


# 邻接矩阵归一化
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.max(1).A + 1e-5, dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)
    return adj


class GraphData(torch.utils.data.Dataset):
    def __init__(self,
                 datareader,
                 fold_id,
                 split):
        self.fold_id = fold_id
        self.split = split
        # self.rnd_state = datareader.rnd_state
        self.set_fold(datareader.data, fold_id)

    def set_fold(self, data, fold_id):
        self.total = len(data['targets'])
        self.N_nodes_max = data['N_nodes_max']
        self.num_classes = data['num_classes']
        self.num_features = data['num_features']
        self.trans_num_features = data['trans_num_features']
        self.code_num_features = data['code_num_features']
        self.idx = data['splits'][fold_id][self.split]
        # print(len(self.idx))
        # use deepcopy to make sure we don't alter objects in folds
        self.labels = copy.deepcopy([data['targets'][i] for i in self.idx])
        self.adj_list = copy.deepcopy([data['adj_list'][i] for i in self.idx])
        self.features_onehot = copy.deepcopy([data['features_onehot'][i] for i in self.idx])
        print('%s: %d/%d' % (self.split.upper(), len(self.labels), len(data['targets'])))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # return [torch.from_numpy(self.features_onehot[index]).float(),  # node_features
        #         torch.from_numpy(self.adj_list[index]).float(),  # adjacency matrix
        #         int(self.labels[index])]
        return [[torch.from_numpy(self.features_onehot[index][0]).float(),
                 torch.from_numpy(self.features_onehot[index][1]).float()],  # node_features
                torch.from_numpy(self.adj_list[index]).float(),  # adjacency matrix
                int(self.labels[index])]




class DataReader():
    '''
    Class to read the txt files containing all data of the dataset.
    Should work for any dataset from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
    '''

    def __init__(self,
                 data_dir,  # folder with txt files
                 rnd_states=None,
                 use_cont_node_attr=True,
                 folds=10,
                 adj_type='unsym',
                 edge_type='V_edge',
                 Isnormalization=False,
                 trans_Normal='all_size',
                 dataset='new_subG_dataset',
                 trans_normal_type='arctan',
                 code_normal_type='arctan',):

        self.data_dir = data_dir  # pre-path
        self.rnd_states = np.random.RandomState() if rnd_states is None else rnd_states
        self.use_cont_node_attr = use_cont_node_attr
        self.folds = folds
        self.adj_type = adj_type
        self.edge_type = edge_type
        self.Isnormalization = Isnormalization
        self.trans_Normal = trans_Normal
        self.dataset = dataset
        self.trans_normal_type = trans_normal_type
        self.code_normal_type = code_normal_type
        files = os.listdir(self.data_dir)
        data = {}
        # graph data
        nodes, graphs = self.read_graph_nodes_relations(
            list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0])
        # data['adj_list'] = self.read_graph_adj_pkl(list(filter(lambda f: f.find(f'{self.edge_type}_A') >= 0, files))[0])
        data['adj_list'] = self.read_graph_adj(list(filter(lambda f: f.find('_A') >= 0, files))[0], nodes, graphs)

        # node label data -- 本实验不用
        data['features'] = None

        node_labels_file = list(filter(lambda f: f.find('node_labels') >= 0, files))
        if len(node_labels_file) == 1:
            data['features'] = self.read_node_features(node_labels_file[0], nodes, graphs, fn=lambda s: int(s.strip()),
                                                       ftype="f")
        else:
            data['features'] = None

        # graph label data
        data['targets'] = np.array(
            self.parse_txt_file(
                list(filter(lambda f: f.find('graph_labels') >= 0 or f.find('graph_attributes') >= 0, files))[0],
                line_parse_fn=lambda s: int(float(s.strip()))))

        # node attribute data
        if self.use_cont_node_attr:
            data['attr'] = self.read_node_features(list(filter(lambda f: f.find('node_attributes') >= 0, files))[0],
                                                   nodes, graphs,
                                                   fn=lambda s: np.array(list(map(float, s.strip().split(',')))))

            data['code_attr'] = self.read_code_features(list(filter(lambda f: f.find(f'_{self.code_normal_type}_{self.Isnormalization}_code_attributes') >= 0, files))[0],
                                                   graphs,
                                                   fn=lambda s: np.array(list(map(float, s.strip().split(',')))),
                                                   ftype='f')



        # graph filter --- 过滤孤立节点图和无边图
        error_graph1 = [i for i, adj in enumerate(data['adj_list']) if len(adj) == 1]
        error_graph2 = [i for i, adj in enumerate(data['adj_list']) if np.sum(adj) == 0]
        error_graph = set(error_graph1 + error_graph2)
        print("[{}] error graph".format(len(error_graph)))
        data['targets'] = np.delete(data['targets'], list(error_graph))
        for idx in error_graph:
            data['adj_list'].pop(idx)  # 删除邻接矩阵Av
            if data['attr'] is not None:
                data['attr'].pop(idx)  # 删除节点属性
            if data['code_attr'] is not None:
                data['code_attr'].pop(idx) # 删除节点属性
            if data['features'] is not None:
                data['features'].pop(idx)  # 删除节点标签

        # 统计各个graph的数据
        features, n_edges, degrees = [], [], []
        for sample_id, adj in enumerate(data['adj_list']):
            N = len(adj)  # number of nodes
            if data['features'] is not None:
                assert N == len(data['features'][sample_id]), (N, len(data['features'][sample_id]))
            if not np.allclose(adj, adj.T):
                # print(sample_id, 'not symmetric')
                pass
            adj_I = copy.deepcopy(adj)
            adj_I[adj > 0] = 1
            n = np.sum(adj_I)  # total sum of edges
            n_edges.append(int(n / 2))  # undirected edges, so need to divide by 2
            degrees.extend(list(np.sum(adj, 1)))
            if data['features'] is not None:
                features.append(np.array(data['features'][sample_id]))

        # Create features over graphs as one-hot vectors for each node
        if data['features'] is not None:
            features_all = np.concatenate(features)
            features_min = features_all.min()
            num_features = int(features_all.max() - features_min + 1)  # number of possible values
        # maxNumNodes = max([len(adj) for adj in data['adj_list']])
        features_onehot = []
        for sample_id, adj in enumerate(data['adj_list']):
            N = adj.shape[0]
            if data['features'] is not None:
                x = data['features'][sample_id]
                feature_onehot = np.zeros((len(x), num_features))
                for node, value in enumerate(x):
                    feature_onehot[node, value - features_min] = 1
                if self.use_cont_node_attr:
                    feature_attr = normalize_features(np.array(data['attr'][sample_id]))
                    try:
                        node_features = np.concatenate((feature_onehot, feature_attr), axis=1)
                        features_onehot.append(node_features)
                    except:
                        raise ("节点数和特征数不匹配：第{}个图，节点数：{}；特征向量数：{}".format(sample_id, N, len(feature_attr)))
                else:
                    features_onehot.append(feature_onehot)
            elif self.use_cont_node_attr:
                # feature_attr = normalize_features(np.array(data['attr'][sample_id]))
                # feature_code_attr = normalize_features(np.array(data['code_attr'][sample_id]))
                feature_attr = np.array(data['attr'][sample_id])
                feature_code_attr = np.array(data['code_attr'][sample_id])

                # features_onehot.append(feature_attr)
                features_onehot.append([feature_attr, feature_code_attr])
            else:
                raise ("没有特征可用！")
        trans_num_features = features_onehot[0][0].shape[1]
        code_num_features = features_onehot[0][1].shape[1]
        shapes = [len(adj) for adj in data['adj_list']]
        labels = data['targets']  # graph class labels
        labels -= np.min(labels)  # to start from 0
        classes = np.unique(labels)
        num_classes = len(classes)

        if not np.all(np.diff(classes) == 1):
            print('making labels sequential, otherwise pytorch might crash')
            labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
            for lbl in range(num_classes):
                labels_new[labels == classes[lbl]] = lbl
            labels = labels_new
            classes = np.unique(labels)
            assert len(np.unique(labels)) == num_classes, np.unique(labels)

        def stats(x):
            return (np.mean(x), np.std(x), np.min(x), np.max(x))

        print('N nodes avg/std/min/max: \t%.2f/%.2f/%d/%d' % stats(shapes))
        print('N edges avg/std/min/max: \t%.2f/%.2f/%d/%d' % stats(n_edges))
        print('Node degree avg/std/min/max: \t%.2f/%.2f/%d/%d' % stats(degrees))
        print('Node trans_old features dim: \t\t%d' % trans_num_features)
        print('Node code features dim: \t\t%d' % code_num_features)
        print('N classes: \t\t\t%d' % num_classes)
        print('Classes: \t\t\t%s' % str(classes))
        for lbl in classes:
            print('Class %d: \t\t\t%d samples' % (lbl, np.sum(labels == lbl)))

        if data['features'] is not None:
            for u in np.unique(features_all):
                print('feature {}, count {}/{}'.format(u, np.count_nonzero(features_all == u), len(features_all)))

        N_graphs = len(labels)  # number of samples (graphs) in data
        assert N_graphs == len(data['adj_list']) == len(features_onehot), 'invalid data'

        data['features_onehot'] = features_onehot
        data['targets'] = labels

        data['N_nodes_max'] = np.max(shapes)  # max number of nodes
        data['trans_num_features'] = trans_num_features
        data['code_num_features'] = code_num_features
        data['num_features'] = trans_num_features
        data['num_classes'] = num_classes

        self.data = data

    def choice_edge_num_spilit_idx_new(self, i, edges_num, proportion):
        # edges_num 子图规模  10，20，30
        np.random.seed(self.rnd_states[i])
        if proportion == '1:n':
            # Todo 特征变了这里就要改
            save_path = f'/public/MountData/DataDir/jj/Ponzi_EWES/Ponzi_detect_model/data_deal/graph_class_data/{self.rnd_states[i]}/'
            train_val_df = pd.read_csv(save_path + '/train_idx.csv')
            train_idx = train_val_df['idx'].tolist()
            val_df = pd.read_csv(save_path + '/val_idx.csv')
            val_idx = val_df['idx'].tolist()

            test_idx_df = pd.read_csv(save_path + f'/edge_num_{edges_num}/test_idx.csv')
            test_idx = test_idx_df['idx'].tolist()

            train_ids = np.array(train_idx)
            val_ids = np.array(val_idx)
            test_ids = np.array(test_idx)

            splits = []
            splits.append({'train': train_ids,
                           'val': val_ids,
                           'test': test_ids})
            self.data['splits'] = splits

    def split_fold(self, i):
        # Create train/test sets first
        train_ids, test_ids = split_ids(self.rnd_states[i].permutation(len(self.data['targets'])), folds=self.folds)
        # Create val sets
        # ts_all = copy.deepcopy(test_ids)
        # val_ids = []
        # test_ids_final = []
        # for ts_set in ts_all:
        #     vl_set = np.random.choice(ts_set,(len(ts_set)//2),replace=False)
        #     val_ids.append(vl_set)
        #     test_ids_final.append(np.array(list(set(ts_set).difference(set(vl_set)))))
        # # val_ids = [np.array(ts_set[:(len(ts_set)//4)]) for ts_set in test_ids]
        # assert len(val_ids)== len(train_ids),"valid set size error:  len(val_ids)={};len(train_ids)={}".format(len(val_ids),len(train_ids))
        splits = []
        for fold in range(len(train_ids)):
            splits.append({'train': train_ids[fold],
                           'val': test_ids[fold][:len(test_ids[fold]) // 2],
                           'test': test_ids[fold][len(test_ids[fold]) // 2:]})
        self.data['splits'] = splits

    def parse_txt_file(self, fpath, line_parse_fn=None):
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
        return data

    def read_graph_adj(self, fpath, nodes, graphs):
        edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
        adj_dict = {}
        for edge in edges:
            node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
            node2 = int(edge[1].strip()) - 1
            graph_id = nodes[node1]
            assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
            if graph_id not in adj_dict:
                n = len(graphs[graph_id])
                adj_dict[graph_id] = np.zeros((n, n))
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]
            assert len(ind1) == len(ind2) == 1, (ind1, ind2)
            adj_dict[graph_id][ind1, ind2] = 1

        adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]

        return adj_list

    def read_graph_adj_pkl(self, fpath):
        adj_list = pkl.load(open(os.path.join(self.data_dir, fpath), 'rb'))
        # adj_list = [normalize_adj(item) for item in adj_list]
        if isinstance(adj_list[0], sp.csr_matrix) or isinstance(adj_list[0], sp.coo_matrix) or isinstance(adj_list[0],
                                                                                                          sp.lil_matrix):
            adj_list = [item.A for item in adj_list]
        if self.adj_type == 'sym':
            # adj_list = [item + item.T for item in adj_list]
            adj_list = [item + item.T + np.diag((np.mean(item, axis=0) + np.mean(item, axis=1)) / 2) for item in
                        adj_list]
        elif self.adj_type == 'i':
            for idx in range(len(adj_list)):
                adj_list[idx] += adj_list[idx].T
                adj_list[idx][adj_list[idx] > 0] = 1
        return adj_list

    def read_graph_nodes_relations(self, fpath):
        """

        :param fpath:
        :return: nodes: dict, node2(node belong to which graph)
                  graphs: dict, graph2(which node belong to this graph(array-like))
        """
        graph_ids = self.parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
        nodes, graphs = {}, {}
        for node_id, graph_id in enumerate(graph_ids):
            if graph_id not in graphs:
                graphs[graph_id] = []
            graphs[graph_id].append(node_id)
            nodes[node_id] = graph_id
        graph_ids = np.unique(list(graphs.keys()))
        for graph_id in graph_ids:
            graphs[graph_id] = np.array(graphs[graph_id])
        return nodes, graphs

    def read_node_features(self, fpath, nodes, graphs, fn):
        node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)
        node_features = {}
        for node_id, x in enumerate(node_features_all):
            graph_id = nodes[node_id]
            if graph_id not in node_features:
                node_features[graph_id] = [None] * len(graphs[graph_id])
            ind = np.where(graphs[graph_id] == node_id)[0]
            assert len(ind) == 1, ind
            assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
            node_features[graph_id][ind[0]] = x
        node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return node_features_lst

    def read_code_features(self,fpath, graphs, fn, ftype):
        if ftype == 'f':
            code_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)
            code_features = {}
            for id_, x in enumerate(code_features_all):
                graph_id = id_ + 1
                if graph_id not in code_features:
                    code_features[graph_id] = [code_features_all[id_]] * len(graphs[graph_id])
            code_features_lst = [code_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
        else:
            code_features_lst = np.vstack(pkl.load(open(os.path.join(self.data_dir, fpath), 'rb')))
            code_features_lst = [code_features_lst[graphs[graph]] for graph in graphs]

        return code_features_lst