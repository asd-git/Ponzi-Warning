
import os
import datetime

import random
import time

from typing import List, Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
import numpy as np
import pandas as pd
import torch
from py2neo import Graph
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from model.Graph_classication.Global_attention import GlobalAttentionNet
from model.Graph_classication.asap import ASAP
from model.Graph_classication.sag_pool import SAGPool
from model.models import MLP, AutoEncoder, GCN_Net, GCN1_Net, GCN_Net_attr, GAT_Net
from model.our_model import All_Concat_model_MLP_GCN_test, \
    All_Concat_model_MLP_SAGPool_test, All_Concat_model_MLP_GA_test, All_Concat_model_MLP_ASAP_test, \
    All_Concat_model_no_MLP_GCN_test, All_Concat_model_no_MLP_GAT_test, All_Concat_model_no_MLP_SAGPool_test, \
    All_Concat_model_no_MLP_GA_test, All_Concat_model_no_MLP_ASAP_test, All_Concat_model_GCN_GCN_test, \
    All_Concat_model_GCN_GAT_test, All_Concat_model_GCN_SAGPool_test, All_Concat_model_GCN_GA_test, \
    All_Concat_model_GCN_ASAP_test, All_Concat_model_GAT_GCN_test, All_Concat_model_GAT_GAT_test, \
    All_Concat_model_GAT_SAGPool_test, All_Concat_model_GAT_GA_test, All_Concat_model_GAT_ASAP_test, \
    All_Concat_model_MLP_GAT_test


def setup_seed(seed):
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed(
        seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(
        seed)  # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

def choice_edge_num_spilit_idx_old(edges_num, subG_dataset, seed):
    # edges_num 子图规模  10，20，30

    # Todo 特征变了这里就要改
    ######################################################################
    # 划分好的 训练集、验证集、测试集
    split_idx_path = f'/public/zjj/jj/data/eth_data/graph_class_data/{seed}'
    train_val_df = pd.read_csv(split_idx_path + '/train_idx.csv')
    train_idx = train_val_df['idx'].tolist()
    val_df = pd.read_csv(split_idx_path + '/val_idx.csv')
    val_idx = val_df['idx'].tolist()

    test_idx_df = pd.read_csv(split_idx_path + f'/edge_num_{edges_num}/test_idx.csv')
    test_idx = test_idx_df['idx'].tolist()

    train_ids = np.array(train_idx)
    val_ids = np.array(val_idx)
    test_ids = np.array(test_idx)
    splits = []
    splits.append({'train': train_ids,
                   'val': val_ids,
                   'test': test_ids})
    return splits


def choice_edge_num_spilit_idx_new(edges_num, subG_dataset, seed):
    # edges_num 子图规模  10，20，30

    # Todo 特征变了这里就要改
    ######################################################################
    # 划分好的 训练集、验证集、测试集
    split_idx_path = f'/public/zjj/jj/data/{subG_dataset}/Dataset_tvt_idx/{seed}/'
    train_val_df = pd.read_csv(split_idx_path + '/train_idx.csv')
    train_idx = train_val_df['idx'].tolist()
    val_df = pd.read_csv(split_idx_path + '/val_idx.csv')
    val_idx = val_df['idx'].tolist()

    test_idx_df = pd.read_csv(split_idx_path + f'/edge_num_{edges_num}/test_idx.csv')
    test_idx = test_idx_df['idx'].tolist()

    train_ids = np.array(train_idx)
    val_ids = np.array(val_idx)
    test_ids = np.array(test_idx)
    splits = []
    splits.append({'train': train_ids,
                   'val': val_ids,
                   'test': test_ids})
    return splits


def neo4j_check_isolate(account, database):
    cypher = "match (n:EOA{name:'" + account + "'}) " \
             + "match (n)-[t:Transaction]-(e:EOA) return e"
    res = database.run(cypher).data()
    # print(res)
    return True if res else False


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

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_results = None

    def __call__(self, val_loss, results):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_results = results
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
            # save best result
            self.best_results = results
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"     INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('     INFO: Early stopping')
                self.early_stop = True



def get_files(dirname, extn, max_files=0):
    """
    获取文件路径
    :param dirname: 指定找的文件夹的路径
    :param extn: 例如 后缀名为'.csv'格式
    :param max_files:可自行定义，可排序找出最大的几个文件
    :return: 返回该文件夹下的指定路径
    """

    all_files = [os.path.join(dirname, f) for f in os.listdir(dirname) if f.endswith(extn)]

    # all_files = []
    # for f in os.listdir(dirname):
    #     if f.split('_')[1] in filename_ls and f.endswith(extn):
    #         file = os.path.join(dirname, f)
    #         all_files.append(file)

    all_files_name = []
    for root, dirs, files in os.walk(dirname):
        for f in files:
            if f.endswith(extn):
                all_files.append(os.path.join(root, f))
                f_name = f.split(".")[0]
                all_files_name.append(f_name)

    all_files = list(set(all_files))
    all_files.sort()
    all_files_name = list(set(all_files_name))
    all_files_name.sort()
    if max_files:
        return all_files[:max_files], all_files_name[:max_files]
    else:
        return all_files, all_files_name


def load_sparse_csr(filename):
    """
    下载并保存为 csr_matrix格式
    :param filename: 文件名
    :return: 解压 csr_matrix格式
    """
    loader = np.load(filename, allow_pickle=True)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def num_to_dic_df(df):
    def get_key(d, value):
        k = [k for k, v in d.items() if v == value]
        return k[0]

    ls_dic = {}

    ls = []
    from_ls = df['from'].tolist()
    to_ls = df['to'].tolist()
    ls.extend(from_ls)
    ls.extend(to_ls)
    ls = list(set(ls))

    for i, node in enumerate(ls):
        ls_dic[i] = node

    df['f_dic'] = df['from'].apply(lambda x: get_key(ls_dic, x))
    df['t_dic'] = df['to'].apply(lambda x: get_key(ls_dic, x))

    # node_num = pd.DataFrame(ls_dic)
    # node_num.to_csv(mkdir('./node-num/PUP_node_num.csv'),index=False)
    # print('完成')

    return df


def duplicates_columns(path, type):
    """
    去除 重复列名
    :param path:
    :param type: ["feature","relation","node_info"]
    :return:
    """
    if type == "feature":
        df1 = pd.read_csv(path)
        feature_res = df1[(df1['Address'] != 'Address')]
        feature_res.to_csv(path, index=False)
    if type == "relation":
        relation_df = pd.read_csv(path)
        relation_res = relation_df[
            (relation_df['from'] != 'from') & (relation_df['to'] != 'to') & (relation_df['values'] != 'values')]
        relation_res.to_csv(path, index=False)

    if type == "node_info":
        node_info_df = pd.read_csv(path)
        node_info_res = node_info_df[
            (node_info_df['Address'] != 'Address') & (node_info_df['type'] != 'type') & (
                    node_info_df['timestamp'] != 'timestamp')]
        node_info_res.to_csv(path, index=False)


def look_lifetime(name, database):
    """
    查看合约的生命周期
    :param name:
    :return:
    """

    def timestamp_to_datetime(timeStamp):
        """
        将时间戳转为日期（日期格式）
        """
        dateArray = datetime.datetime.fromtimestamp(timeStamp)
        otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
        # 将 string => datetime
        date = datetime.datetime.strptime(otherStyleTime, '%Y-%m-%d %H:%M:%S')
        return date

    trans_cyber = "match (s)-[r:`trans_old`]-(e:CA{name:'" + name + "'}) " \
                                                                "return startNode(r).name as from,endNode(r).name as to, r.value as value, r.timestamp as timestamp order by timestamp DESC Limit 1"
    trans_res = pd.DataFrame(database.run(trans_cyber).data())

    create_cyber = "match (s)-[r:`create`]->(e:CA{name:'" + name + "'}) " \
                                                                   "return startNode(r).name as from,endNode(r).name as to, r.value as value, r.timestamp as timestamp order by timestamp Limit 1"
    create_res = pd.DataFrame(database.run(create_cyber).data())
    last_datetime = timestamp_to_datetime(int(trans_res['timestamp'].values[0]))
    create_datetime = timestamp_to_datetime(int(create_res['timestamp'].values[0]))
    lifetime = last_datetime - create_datetime

    return lifetime.days


# 判断地址的类型(EOA/CA)
def nodetype(name, database):
    """
    判断地址的类型(EOA/CA)
    :param name:
    :return:
    """
    EOA_cyber1 = "match(n:EOA{name:'" + name + "'})return labels(n)"
    SC_cyber1 = "match(n:CA{name:'" + name + "'})return labels(n)"

    EOA_type = database.run(EOA_cyber1).data()
    CA_type = database.run(SC_cyber1).data()

    node_type = None
    if (EOA_type == []) & (CA_type == []):
        print(name)

    elif EOA_type == []:
        node_type = CA_type[0]['labels(n)'][0]
        # print(CA_type[0]['labels(n)'][0])
    elif CA_type == []:
        node_type = EOA_type[0]['labels(n)'][0]
        # print(EOA_type[0]['labels(n)'][0])
    return node_type


def gini(p):
    "https://blog.csdn.net/lly1122334/article/details/104253254"
    if p != []:
        cum = np.cumsum(sorted(np.append(p, 0)))
        sum = cum[-1]
        x = np.array(range(len(cum))) / len(p)
        y = cum / sum
        B = np.trapz(y, x=x)
        A = 0.5 - B
        G = A / (A + B)
    else:
        G = 0
    return G


def contract_trans_info(node, trans_res_edges_df, lifetime):
    """
    提取节点的交易特征
    :param node:
    :param trans_res_edges_df:
    :return:
    """
    inv_value_ls = []
    return_value_ls = []
    V_maxinv = 0
    V_maxreturn = 0
    N_inv = 0
    N_return = 0
    V_inv = 0
    V_return = 0
    bal = 0
    paid_rate = 0
    temp = []
    trans_list = []
    temp.append(node)
    inv_adr = set()
    df_node_from = trans_res_edges_df[trans_res_edges_df['to'] == node]
    shape = df_node_from.shape[0]
    for i in range(shape):
        data = df_node_from.iloc[i, :]
        if data['values'] > 0:
            N_inv = N_inv + 1  # 合约收到投资总次数
            V_inv = V_inv + data['values']  # 合约收到投资总金额
            cal_value = data['values']
            inv_value_ls.append(cal_value)  # 投资的金额放置，计算方差
            if V_maxinv <= data['values']:
                V_maxinv = data['values']
        if data['from'] not in inv_adr:
            inv_adr.add(data['from'])  # 所有进行投资的人

    return_adr = set()
    df_node_to = trans_res_edges_df[trans_res_edges_df['from'] == node]
    shape = df_node_to.shape[0]
    for i in range(shape):
        data = df_node_to.iloc[i, :]
        if data['values'] > 0:
            N_return = N_return + 1
            V_return = V_return + data['values']  # 合约进行回报的金额数
            cal_value = data['values']
            return_value_ls.append(cal_value)  # 回报的金额放置，计算方差
            if V_maxreturn <= data['values']:
                V_maxreturn = data['values']  # 投资者收到的最大一笔回报
            if data['to'] not in return_adr:
                return_adr.add(data['to'])  # 所有收到回报的人

    bal = V_inv - V_return  # 合约余额
    count = 0
    for j in inv_adr:
        if j in return_adr:
            count = count + 1
    if len(inv_adr):
        paid_rate = count / len(inv_adr)  # 投资者收到至少一笔回报的比例
    else:
        paid_rate = 0

    bal = bal  # 将单位改成以太
    V_maxinv = V_maxinv
    V_maxreturn = V_maxreturn
    V_inv = V_inv
    V_return = V_return

    V_meaninv = np.mean(inv_value_ls)
    V_meanreturn = np.mean(return_value_ls)

    V_Gini_inv = gini(inv_value_ls)
    V_Gini_return = gini(return_value_ls)

    V_stdinv = np.std(inv_value_ls)
    V_stdreturn = np.std(return_value_ls)

    temp.append(bal)  # 余额
    temp.append(V_inv)  # 投资总金额
    temp.append(V_return)  # 回报总金额
    temp.append(V_meaninv)  # 投资均值
    temp.append(V_meanreturn)  # 回报均值
    temp.append(V_stdinv)  # 投资方差值
    temp.append(V_stdreturn)  # 回报方差值
    temp.append(V_maxinv)  # 最大投资额
    temp.append(V_maxreturn)  # 最大回报值
    temp.append(N_inv)  # 投资总次数
    temp.append(N_return)  # 付款总次数
    temp.append(V_Gini_inv)  # 投资基尼系数
    temp.append(V_Gini_return)  # 回报基尼系数
    temp.append(lifetime)  # 生命周期
    temp.append(paid_rate)  # 投资者收到至少一笔回报的比例

    trans_list.append(temp)
    return trans_list


def extract_contract_trans_info(trans_res, name, node_lifetime, savepath):
    """
    分类合约的本身信息(特征、关系)
    :param trans_res:
    :param name:
    :param edge_nums:
    :param type_:
    :param savepath:
    :return:
    """
    if trans_res.shape != (0, 0):
        numbers = [float(x) for x in trans_res['value'].tolist()]
        trans_res['values'] = numbers
        # 子图边关系提取
        # print(f"Contract relation({name})...")

        trans_res_edges_df = trans_res

        trans_relation_df = trans_res_edges_df[['from', 'to', 'values']]

        sum_count_res = trans_relation_df.groupby(['from', 'to'])['values'].agg(['count', 'sum']).reset_index()
        # sum_count_df = num_to_dic_df(sum_count_res)

        save_dir = mkdir(savepath + f"/contract_relation/")
        trans_relation_df.to_csv(save_dir + f"/{name}_relation.csv", index=False)

        # 子图交易特征提取
        # print(f"Contract  feature({name})...")
        trans_info_ls = contract_trans_info(name, trans_res_edges_df, node_lifetime)
        col_name = ['Address', 'balance', 'V_invest', 'V_return', 'V_mean_invest', 'V_mean_return',
                    'V_std_invest',
                    'V_std_return',
                    'V_max_inv', 'V_max_return', 'N_invest', 'N_return', 'V_Gini_invest', 'V_Gini_return',
                    'lifetime', 'paid_rate']
        trans_feature_df = pd.DataFrame(data=trans_info_ls, columns=col_name)
        trans_feature_df = trans_feature_df.fillna(0)
        save_dir = mkdir(savepath + f"/contract_feature/")
        trans_feature_df.to_csv(save_dir + f"/{name}_feature.csv", index=False)
        return sum_count_res, trans_feature_df


def extract_node_1k_trans_info(trans_res, name, node_lifetime, database, savepath):
    """
     提取合约一阶邻居的信息(节点类型信息、特征、关系)
    :param trans_res:
    :param name:
    :param edge_nums:
    :param type_:
    :param savepath:
    :return:
    """

    node_1k_info = 'node_1k_info'
    node_1k_relation = 'node_1k_relation'
    node_1k_feature = 'node_1k_feature'
    if trans_res.shape != (0, 0):
        numbers = [float(x) for x in trans_res['value'].tolist()]
        trans_res['values'] = numbers
        # 子图边关系提取
        trans_res_edges_df = trans_res
        sum_count_res = trans_res_edges_df.groupby(['from', 'to'])

        # 前edge_nums条边的 最新的时间戳的邻居节点提取
        # print(f"Contract({name}) node_1k  extract...")
        node_timestamp_info = {}
        for i in list(sum_count_res):
            group_df = i[1]
            last_time_df = group_df.iloc[-1, :][['from', 'to', 'timestamp']]

            if last_time_df['from'] == name:
                nb_1k_node = last_time_df['to']
                timestamp = int(last_time_df['timestamp'])
                if nb_1k_node in node_timestamp_info.keys():
                    if timestamp > node_timestamp_info[nb_1k_node]:
                        node_timestamp_info[nb_1k_node] = timestamp
                    else:
                        continue
                else:
                    node_timestamp_info[nb_1k_node] = timestamp
            if last_time_df['to'] == name:
                nb_1k_node = last_time_df['from']
                timestamp = int(last_time_df['timestamp'])
                if nb_1k_node in node_timestamp_info.keys():
                    if timestamp > node_timestamp_info[nb_1k_node]:
                        node_timestamp_info[nb_1k_node] = timestamp
                    else:
                        continue
                else:
                    node_timestamp_info[nb_1k_node] = timestamp
        """
        node_1k_info_ls ：一阶节点的信息：[节点名称、节点类别、最新时间戳]
        """
        # print(f"node_1k: start extractContract({name}) node_1k ...")
        node_1k_info_ls = [[key, nodetype(key, database), str(node_timestamp_info[key])] for key in
                           node_timestamp_info.keys()]
        node_1k_info_df = pd.DataFrame(data=node_1k_info_ls, columns=['Address', 'type', 'timestamp'])
        # 保存 合约一阶邻居的信息

        new_save_dir = mkdir(savepath + f"/node_1k/")
        save_dir = mkdir(savepath + f"/node_1k/{node_1k_info}/")
        node_1k_info_df.to_csv(save_dir + f"/{name}_{node_1k_info}.csv", index=False)
        node_1k_info_df.to_csv(new_save_dir + f"/{node_1k_info}.csv", index=False, mode='a')

        node_1k_feature_ls = []

        for index, item in enumerate(node_1k_info_ls):
            node_name = item[0]
            node_type = item[1]
            node_last_timestamp = item[2]
            # print(node_name, node_type, node_last_timestamp)
            trans_1K_cyber = "match (s)-[r:`trans_old`]-(e:" + node_type + "{name:'" + node_name + "'}) " + " where r.timestamp <= '" + node_last_timestamp + "' " + \
                             "return startNode(r).name as from,endNode(r).name as to, r.value as value, r.timestamp as timestamp order by timestamp"

            trans_1K_res = pd.DataFrame(database.run(trans_1K_cyber).data())

            if trans_1K_res.shape != (0, 0):
                numbers = [float(x) for x in trans_1K_res['value'].tolist()]
                trans_1K_res['values'] = numbers
                # 一阶的邻居关系  # trans_1K_res: 1阶邻居在timestamp之前的 网络关系
                trans_1K_res = trans_1K_res[['from', 'to', 'values']]
                save_dir = mkdir(savepath + f"/node_1k/{node_1k_relation}/")
                trans_1K_res.to_csv(save_dir + f"/{name}_{node_1k_relation}.csv", index=False, mode='a')
                trans_1K_res.to_csv(new_save_dir + f"/{node_1k_relation}.csv", index=False, mode='a')

                # 一阶邻居特征的提取

                # print(f"{index + 1}/{len(node_1k_info_ls)}", )
                trans_1K_info_ls = contract_trans_info(node_name, trans_1K_res, node_lifetime)
                col_name = ['Address', 'balance', 'V_invest', 'V_return', 'V_mean_invest', 'V_mean_return',
                            'V_std_invest',
                            'V_std_return',
                            'V_max_inv', 'V_max_return', 'N_invest', 'N_return', 'V_Gini_invest', 'V_Gini_return',
                            'lifetime', 'paid_rate']
                node_1k_feature_df = pd.DataFrame(data=trans_1K_info_ls, columns=col_name)
                node_1k_feature_df = node_1k_feature_df.fillna(0)
                node_1k_feature_ls.append(node_1k_feature_df)
                save_dir = mkdir(savepath + f"/node_1k/{node_1k_feature}/")
                # 单个保存
                node_1k_feature_df.to_csv(save_dir + f"/{name}_{node_1k_feature}.csv", index=False, mode='a')
                # 整体保存
                node_1k_feature_df.to_csv(new_save_dir + f"/{node_1k_feature}.csv", index=False, mode='a')
        contract_trans_1k_feature = pd.concat(node_1k_feature_ls, ignore_index=True)

        # print(f"node_1k: {type_} Contract({name}) node_1k_info ...")
        # node_info_1k_path = new_save_dir + f"/{type_}_node_1k_info.csv"
        # duplicates_columns(node_info_1k_path, 'node_info')
        #
        # print(f"node_1k: {type_} Contract({name}) node_1k_relation ...")
        # relation_1k_path = new_save_dir + f"/{type_}_node_1k_relation.csv"
        # duplicates_columns(relation_1k_path, 'relation')
        # relation_1k_name_path = savepath + f"/node_1k/{node_1k_relation}/" + f"/{type_}_{name}_{node_1k_relation}.csv"
        # duplicates_columns(relation_1k_name_path, 'relation')
        #
        # print(f"node_1k: {type_} Contract({name}) node_1k_feature ...")
        # feature_1k_path = new_save_dir + f"/{type_}_node_1k_feature.csv"
        # duplicates_columns(feature_1k_path, "feature")
        # feature_1k_name_path = savepath + f"/node_1k/{node_1k_feature}/" + f"/{type_}_{name}_{node_1k_feature}.csv"
        # duplicates_columns(feature_1k_name_path, 'feature')

        return contract_trans_1k_feature


def extract_node_code_info(account):
    """
    获得合约的代码特征
    :account:
    :return:
    """

    code_df = pd.read_csv('.././data/PUP_node_dataset/Opcode_net_PUP.csv')
    code_fea = code_df[code_df['Contract'] == account]
    return code_fea


def kfold_split_idx(edge_num_ls, kfold, seed):
    "无验证集"
    ##########################################
    # 生成kfold_idx  训练集 测试集索引
    # PUP_node = pd.read_csv('./data/PUP_node_dataset/PUP_100up/PUP_node(400).csv')
    PUP_node = pd.read_csv('D:\浙工大研究生生活\区块链\A 以太坊庞氏骗局预警评估系统 论文\Ponzi_ieee\data\PUP_node_dataset\PUP_100up\PUP_node(400).csv')
    pos = PUP_node[PUP_node['label'] == 1]['Contract'].tolist()  # 75个
    neg = PUP_node[PUP_node['label'] == 0]['Contract'].tolist()
    pos_len = len(pos)  # 75
    neg_len = len(neg)  # 325
    all_len = pos_len + neg_len  # 400
    m = np.array(list(range(0, all_len)))
    n = np.array([1] * pos_len + [0] * neg_len)
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=kfold, shuffle=True)
    kfold_train_test = []
    for train_index, test_index in skf.split(m, n):
        kfold_train_test.append([train_index, test_index])

    kfold_index_dic = dict()
    for i, item in enumerate(kfold_train_test):
        if i not in kfold_index_dic.keys():
            kfold_index_dic[i] = dict()
            train_index, test_index = item[0], item[1]
            train_all_idx = [j + (all_len * i) for i in range(len(edge_num_ls)) for j in train_index]
            kfold_index_dic[i]["train_idx"] = train_all_idx
            kfold_index_dic[i]["test_idx"] = dict()
            for edge_num in range(len(edge_num_ls)):
                test_edge_idx = []
                for j in test_index:
                    test_edge_idx.append(j + (all_len * edge_num))

                kfold_index_dic[i]["test_idx"][(edge_num + 1) * 10] = test_edge_idx

    #########################################################################
    return kfold_index_dic

def diff_index(reals, preds):
    diff_index = [index for index, (item1, item2) in enumerate(zip(reals, preds)) if item1 != item2]
    true_label = [reals[index] for index in diff_index]

    return diff_index,true_label

def kfold_split_tvt_idx(edge_num_ls, kfold, seed):
    "有验证集"
    ##########################################
    # 生成kfold_idx  训练集 测试集索引
    # todo
    # PUP_node = pd.read_csv('./data/PUP_node_dataset/PUP_100up/PUP_node(400).csv')
    PUP_node = pd.read_csv('D:\浙工大研究生生活\区块链\A 以太坊庞氏骗局预警评估系统 论文\Ponzi_ieee\data\PUP_node_dataset\PUP_100up\PUP_node(400).csv')
    pos = PUP_node[PUP_node['label'] == 1]['Contract'].tolist()  # 75个
    neg = PUP_node[PUP_node['label'] == 0]['Contract'].tolist()
    pos_len = len(pos)  # 75
    neg_len = len(neg)  # 325
    all_len = pos_len + neg_len  # 400
    m = np.array(list(range(0, all_len)))
    n = np.array([1] * pos_len + [0] * neg_len)
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=kfold, shuffle=True)
    Kfold_train_val_test = []
    for train_index, test_index in skf.split(m, n):
        train_idx = train_index
        test_idx = test_index
        np.random.seed(seed)
        val_idx = np.random.choice(train_idx, int(len(train_idx) * 0.2),replace=False)
        train_idx_ = np.array([i for i in train_idx if i not in val_idx])
        Kfold_train_val_test.append([train_idx_, val_idx, test_idx])

    kfold_index_dic = dict()
    for i, item in enumerate(Kfold_train_val_test):
        if i not in kfold_index_dic.keys():
            kfold_index_dic[i] = dict()
            train_idxs, val_idxs, test_idxs = item[0], item[1], item[2]
            train_all_idx = [j + (all_len * i) for i in range(len(edge_num_ls)) for j in train_idxs]
            val_all_idx = [j + (all_len * i) for i in range(len(edge_num_ls)) for j in val_idxs]
            kfold_index_dic[i]["train_idx"] = train_all_idx
            kfold_index_dic[i]["val_idx"] = val_all_idx
            kfold_index_dic[i]["test_idx"] = dict()
            for edge_num in range(len(edge_num_ls)):
                test_edge_idx = []
                for j in test_idxs:
                    test_edge_idx.append(j + (all_len * edge_num))

                kfold_index_dic[i]["test_idx"][(edge_num + 1) * 10] = test_edge_idx

    #########################################################################
    return kfold_index_dic

def kfold_split_tvt_val_idx(edge_num_ls, kfold, seed):
    "有验证集,验证集也按时序划分好"
    ##########################################
    # 生成kfold_idx  训练集 测试集索引
    # todo
    PUP_node = pd.read_csv('./data/PUP_node_dataset/PUP_100up/PUP_node(400).csv')
    # PUP_node = pd.read_csv('D:\浙工大研究生生活\区块链\A 以太坊庞氏骗局预警评估系统 论文\Ponzi_ieee\data\PUP_node_dataset\PUP_100up\PUP_node(400).csv')
    pos = PUP_node[PUP_node['label'] == 1]['Contract'].tolist()  # 75个
    neg = PUP_node[PUP_node['label'] == 0]['Contract'].tolist()
    pos_len = len(pos)  # 75
    neg_len = len(neg)  # 325
    all_len = pos_len + neg_len  # 400
    m = np.array(list(range(0, all_len)))
    n = np.array([1] * pos_len + [0] * neg_len)
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=kfold, shuffle=True)
    Kfold_train_val_test = []
    for train_index, test_index in skf.split(m, n):
        train_idx = train_index
        test_idx = test_index
        np.random.seed(seed)
        val_idx = np.random.choice(train_idx, int(len(train_idx) * 0.2),replace=False)
        train_idx_ = np.array([i for i in train_idx if i not in val_idx])
        Kfold_train_val_test.append([train_idx_, val_idx, test_idx])

    kfold_index_dic = dict()
    for i, item in enumerate(Kfold_train_val_test):
        if i not in kfold_index_dic.keys():
            kfold_index_dic[i] = dict()
            train_idxs, val_idxs, test_idxs = item[0], item[1], item[2]
            train_all_idx = [j + (all_len * i) for i in range(len(edge_num_ls)) for j in train_idxs]
            # val_all_idx = [j + (all_len * i) for i in range(len(edge_num_ls)) for j in val_idxs]
            kfold_index_dic[i]["train_idx"] = train_all_idx
            kfold_index_dic[i]["val_idx"] = dict()
            kfold_index_dic[i]["test_idx"] = dict()
            for edge_num in range(len(edge_num_ls)):
                val_edge_idx = []
                test_edge_idx = []
                for m in val_idxs:
                    val_edge_idx.append(m + (all_len * edge_num))

                for n in test_idxs:
                    test_edge_idx.append(n + (all_len * edge_num))

                kfold_index_dic[i]["val_idx"][(edge_num + 1) * 10] = val_edge_idx
                kfold_index_dic[i]["test_idx"][(edge_num + 1) * 10] = test_edge_idx

    #########################################################################
    return kfold_index_dic



def kfold_split_tvt_final_idx(edge_num_ls, kfold, seed):
    "有验证集   只取最终的索引"
    ##########################################
    # 生成kfold_idx  训练集 测试集索引
    # todo
    # PUP_node = pd.read_csv('./data/PUP_node_dataset/PUP_100up/PUP_node(400).csv')
    PUP_node = pd.read_csv('D:\浙工大研究生生活\区块链\A 以太坊庞氏骗局预警评估系统 论文\Ponzi_ieee\data\PUP_node_dataset\PUP_100up\PUP_node(400).csv')
    pos = PUP_node[PUP_node['label'] == 1]['Contract'].tolist()  # 75个
    neg = PUP_node[PUP_node['label'] == 0]['Contract'].tolist()
    pos_len = len(pos)  # 75
    neg_len = len(neg)  # 325
    all_len = pos_len + neg_len  # 400
    m = np.array(list(range(0, all_len)))
    n = np.array([1] * pos_len + [0] * neg_len)
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=kfold, shuffle=True)
    Kfold_train_val_test = []
    for train_index, test_index in skf.split(m, n):
        train_idx = train_index
        test_idx = test_index
        np.random.seed(seed)
        val_idx = np.random.choice(train_idx, int(len(train_idx) * 0.2),replace=False)
        train_idx_ = np.array([i for i in train_idx if i not in val_idx])
        Kfold_train_val_test.append([train_idx_, val_idx, test_idx])

    kfold_index_dic = dict()
    for i, item in enumerate(Kfold_train_val_test):
        if i not in kfold_index_dic.keys():
            kfold_index_dic[i] = dict()
            train_idxs, val_idxs, test_idxs = item[0], item[1], item[2]
            train_all_idx = [j + (all_len * (len(edge_num_ls)-1)) for j in train_idxs]
            val_all_idx = [j + (all_len * (len(edge_num_ls)-1)) for j in val_idxs]
            kfold_index_dic[i]["train_idx"] = train_all_idx
            kfold_index_dic[i]["val_idx"] = val_all_idx
            kfold_index_dic[i]["test_idx"] = dict()
            for edge_num in range(len(edge_num_ls)):
                test_edge_idx = []
                for j in test_idxs:
                    test_edge_idx.append(j + (all_len * (len(edge_num_ls)-1)))

                kfold_index_dic[i]["test_idx"][(edge_num + 1) * 10] = test_edge_idx

    #########################################################################
    return kfold_index_dic


def kfold_split_dy_tvt_idx(edge_num_ls, kfold, seed):
    "动态有验证集"
    ##########################################
    # 生成kfold_idx  训练集 测试集索引
    # todo
    # PUP_node = pd.read_csv('./data/PUP_node_dataset/PUP_100up/PUP_node(400).csv')
    PUP_node = pd.read_csv('D:\浙工大研究生生活\区块链\A 以太坊庞氏骗局预警评估系统 论文\Ponzi_ieee\data\PUP_node_dataset\PUP_100up\PUP_node(400).csv')
    pos = PUP_node[PUP_node['label'] == 1]['Contract'].tolist()  # 75个
    neg = PUP_node[PUP_node['label'] == 0]['Contract'].tolist()
    pos_len = len(pos)  # 75
    neg_len = len(neg)  # 325
    all_len = pos_len + neg_len  # 400
    m = np.array(list(range(0, all_len)))
    n = np.array([1] * pos_len + [0] * neg_len)
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=kfold, shuffle=True)
    Kfold_train_val_test = []
    for train_index, test_index in skf.split(m, n):
        train_idx = train_index
        test_idx = test_index
        np.random.seed(seed)
        val_idx = np.random.choice(train_idx, int(len(train_idx) * 0.2),replace=False)
        train_idx_ = np.array([i for i in train_idx if i not in val_idx])
        Kfold_train_val_test.append([train_idx_, val_idx, test_idx])

    kfold_index_dic = dict()
    for i, item in enumerate(Kfold_train_val_test):
        if i not in kfold_index_dic.keys():
            kfold_index_dic[i] = dict()
            train_idxs, val_idxs, test_idxs = item[0], item[1], item[2]

            kfold_index_dic[i]["train_idx"] = train_idxs.tolist()
            kfold_index_dic[i]["val_idx"] = val_idxs.tolist()
            kfold_index_dic[i]["test_idx"] = test_idxs.tolist()


    #########################################################################
    return kfold_index_dic


def load_data_machine(edge_num, edge_num_ls, rnd_state, kfold):
    """
    有 交易  代码 特征
    """

    train_test_kfold_dataset = []

    # data = kfold_split_idx(edge_num_ls, kfold, rnd_state)
    data = kfold_split_tvt_idx(edge_num_ls, kfold, rnd_state)

    root_dir = '.././data1/eth/pup/To_Machine/'
    trans_atrributes = root_dir + 'CA_trans_feature.csv'
    trans_df = pd.read_csv(trans_atrributes)
    trans_arctan = trans_df.iloc[:, 1:-1].apply(lambda x: np.arctan(x) * (2 / np.pi))  # 非线性归一化

    code_atrributes = root_dir + 'CA_code_feature.csv'
    code_df = pd.read_csv(code_atrributes)
    code_std = code_df.iloc[:, 2:].apply(lambda x: (x - np.mean(x)) / np.std(x))
    for kf in range(kfold):
        train_train = trans_arctan[trans_arctan.index.isin(data[kf]['train_idx'])]
        train_test = trans_arctan[trans_arctan.index.isin(data[kf]['test_idx'][edge_num])]

        X_trans_train = train_train.to_numpy()
        X_trans_test = train_test.to_numpy()

        Y_trans_train = trans_df[trans_df.index.isin(data[kf]['train_idx'])].iloc[:, -1].tolist()
        Y_trans_test = trans_df[trans_df.index.isin(data[kf]['test_idx'][edge_num])].iloc[:, -1].tolist()

        train_trans_x = torch.tensor(X_trans_train)
        train_trans_y = torch.LongTensor(Y_trans_train)

        test_trans_x = torch.tensor(X_trans_test)
        test_trans_y = torch.LongTensor(Y_trans_test)
        ###########################################################################################################

        # code

        code_train = code_std[code_std.index.isin(data[kf]['train_idx'])]

        code_test = code_std[code_std.index.isin(data[kf]['test_idx'][edge_num])]

        X_code_train = code_train.to_numpy()
        X_code_test = code_test.to_numpy()

        Y_code_train = code_df[code_df.index.isin(data[kf]['train_idx'])].iloc[:, 1].tolist()
        Y_code_test = code_df[code_df.index.isin(data[kf]['test_idx'][edge_num])].iloc[:, 1].tolist()

        train_code_x = torch.tensor(X_code_train)
        train_code_y = torch.LongTensor(Y_code_train)
        test_code_x = torch.tensor(X_code_test)
        test_code_y = torch.LongTensor(Y_code_test)

        train_x = [train_trans_x, train_code_x.float()]
        test_x = [test_trans_x, test_code_x.float()]
        train_y = train_code_y
        test_y = test_code_y
        train_test_kfold_dataset.append([(train_x, train_y), (test_x, test_y)])

    return train_test_kfold_dataset


def load_data_machine_new(edge_num, edge_num_ls, rnd_state, kfold):
    """
    有 交易
    改了代码 特征
    """

    train_test_kfold_dataset = []

    # data = kfold_split_idx(edge_num_ls, kfold, rnd_state)
    data = kfold_split_tvt_idx(edge_num_ls, kfold, rnd_state)

    root_dir = '.././data_new/eth/pup/To_Machine/'
    trans_atrributes = root_dir + 'CA_trans_feature.csv'
    trans_df = pd.read_csv(trans_atrributes)
    trans_arctan = trans_df.iloc[:, 1:-1].apply(lambda x: np.arctan(x) * (2 / np.pi))  # 非线性归一化

    code_df = pd.read_csv(r'D:\浙工大研究生生活\区块链\A 以太坊庞氏骗局预警评估系统 论文\Ponzi_ieee\data_new\eth\raw\eth_code_attributes.txt',header=None)
    label = [1] * 75 + [0] * 325
    labels = []
    for i in range(len(edge_num_ls)):
        labels.extend(label)

    code_df['label'] = labels
    code_normal = code_df.iloc[:, :-1]

    for kf in range(kfold):
        train_train = trans_arctan[trans_arctan.index.isin(data[kf]['train_idx'])]
        train_test = trans_arctan[trans_arctan.index.isin(data[kf]['test_idx'][edge_num])]

        X_trans_train = train_train.to_numpy()
        X_trans_test = train_test.to_numpy()

        Y_trans_train = trans_df[trans_df.index.isin(data[kf]['train_idx'])].iloc[:, -1].tolist()
        Y_trans_test = trans_df[trans_df.index.isin(data[kf]['test_idx'][edge_num])].iloc[:, -1].tolist()

        train_trans_x = torch.tensor(X_trans_train)
        train_trans_y = torch.LongTensor(Y_trans_train)

        test_trans_x = torch.tensor(X_trans_test)
        test_trans_y = torch.LongTensor(Y_trans_test)
        ###########################################################################################################

        "code"
        code_train = code_normal[code_normal.index.isin(data[kf]['train_idx'])]
        code_test = code_normal[code_normal.index.isin(data[kf]['test_idx'][edge_num])]

        X_code_train = code_train.to_numpy()
        X_code_test = code_test.to_numpy()

        Y_code_train = code_df[code_df.index.isin(data[kf]['train_idx'])].iloc[:, -1].tolist()
        Y_code_test = code_df[code_df.index.isin(data[kf]['test_idx'][edge_num])].iloc[:, -1].tolist()

        train_code_x = torch.tensor(X_code_train)
        train_code_y = torch.LongTensor(Y_code_train)
        test_code_x = torch.tensor(X_code_test)
        test_code_y = torch.LongTensor(Y_code_test)

        train_x = [train_trans_x, train_code_x.float()]
        test_x = [test_trans_x, test_code_x.float()]
        train_y = train_code_y
        test_y = test_code_y
        train_test_kfold_dataset.append([(train_x, train_y), (test_x, test_y)])

    return train_test_kfold_dataset

def load_data_machine_all_feature(edge_num, edge_num_ls, rnd_state, kfold):
    """
    交易和代码特征合并
    """

    train_test_kfold_dataset = []

    # data = kfold_split_idx(edge_num_ls, kfold, rnd_state)
    data = kfold_split_tvt_idx(edge_num_ls, kfold, rnd_state)

    root_dir = '.././data_new/eth/pup/To_Machine/'
    trans_atrributes = root_dir + 'CA_trans_feature.csv'
    trans_df = pd.read_csv(trans_atrributes)
    trans_arctan = trans_df.iloc[:, 1:-1].apply(lambda x: np.arctan(x) * (2 / np.pi))  # 非线性归一化

    code_df = pd.read_csv(r'D:\浙工大研究生生活\区块链\A 以太坊庞氏骗局预警评估系统 论文\Ponzi_ieee\data_new\eth\raw\eth_code_attributes.txt',header=None)
    # code_df = code_df.apply(lambda x: (x - np.min(x)) / np.max(x) - np.min(x))
    code_df = code_df.apply(lambda x: np.arctan(x) * (2 / np.pi))
    label = [1] * 75 + [0] * 325
    labels = []
    for i in range(len(edge_num_ls)):
        labels.extend(label)

    code_df['label'] = labels
    code_normal = code_df.iloc[:, :-1]

    for kf in range(kfold):
        train_train = trans_arctan[trans_arctan.index.isin(data[kf]['train_idx'])]
        train_test = trans_arctan[trans_arctan.index.isin(data[kf]['test_idx'][edge_num])]

        X_trans_train = train_train.to_numpy()
        X_trans_test = train_test.to_numpy()

        Y_trans_train = trans_df[trans_df.index.isin(data[kf]['train_idx'])].iloc[:, -1].tolist()
        Y_trans_test = trans_df[trans_df.index.isin(data[kf]['test_idx'][edge_num])].iloc[:, -1].tolist()

        train_trans_x = torch.tensor(X_trans_train)
        train_trans_y = torch.LongTensor(Y_trans_train)

        test_trans_x = torch.tensor(X_trans_test)
        test_trans_y = torch.LongTensor(Y_trans_test)
        ###########################################################################################################

        "code"
        code_train = code_normal[code_normal.index.isin(data[kf]['train_idx'])]
        code_test = code_normal[code_normal.index.isin(data[kf]['test_idx'][edge_num])]

        X_code_train = code_train.to_numpy()
        X_code_test = code_test.to_numpy()

        Y_code_train = code_df[code_df.index.isin(data[kf]['train_idx'])].iloc[:, -1].tolist()
        Y_code_test = code_df[code_df.index.isin(data[kf]['test_idx'][edge_num])].iloc[:, -1].tolist()

        train_code_x = torch.tensor(X_code_train)
        train_code_y = torch.LongTensor(Y_code_train)
        test_code_x = torch.tensor(X_code_test)
        test_code_y = torch.LongTensor(Y_code_test)

        train_x = [train_trans_x, train_code_x.float()]
        test_x = [test_trans_x, test_code_x.float()]
        train_all_x = torch.cat(train_x,1)
        test_all_x = torch.cat(test_x,1)

        train_y = train_code_y
        test_y = test_code_y
        train_test_kfold_dataset.append([(train_all_x, train_y), (test_all_x, test_y)])

    return train_test_kfold_dataset

class ColumnNormalizeFeatures(BaseTransform):
    r"""Row-normalizes the attributes given in :obj:`attrs` to sum-up to one.

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
    """
    def __init__(self, attrs: List[str] = ["x"]):
        self.attrs = attrs

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                value.div_(value.sum(dim=-1, keepdim=True).clamp_(min=1.))  # change
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


def set_all_random(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # gpu index
    # 设定随机种子，实验结果可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    rnd_states = [
        args.seed - 111,
    ]
    return rnd_states

def print_args(args):
    "打印实验参数"
    # print parameters
    print("==========Parameters=========")
    for arg in vars(args):
        print(arg, getattr(args, arg))
        print("-----------------------------")
    print("=============================")

def get_date():
    return time.strftime("%m-%d", time.localtime())


def model_choice(args,dataset):

    """code"""
    if args.model_type == 'MLP':
        model = MLP(input=dataset.num_code_attributes,
                    dim=args.code_hidden,
                    output=dataset.num_classes).to(args.device)
    if args.model_type == 'AutoEncoder':
        model = AutoEncoder(input=dataset.num_code_attributes,
                            dim=args.code_hidden,
                            output=dataset.num_classes).to(args.device)
    """trans"""
    if args.model_type == 'GCN_Net':
        model = GCN_Net(
            in_channels=dataset.num_features,
            dim=args.trans_hidden,
            out_channels=dataset.num_classes,
            num_layers=args.num_layers,
            pooling=args.pooling,
            dropout=args.dropout
        ).to(args.device)
    if args.model_type == 'GAT_Net':
        model = GAT_Net(
            in_channels=dataset.num_features,
            dim=args.trans_hidden,
            out_channels=dataset.num_classes,
            num_layers=args.num_layers,
            pooling=args.pooling,
            dropout=args.dropout)
    if args.model_type == 'SAGPool':
        model = SAGPool(
            in_channels=dataset.num_features,
            hidden=args.trans_hidden,
            out_channels=dataset.num_classes,
            num_layers=args.num_layers,
            pooling=args.pooling,
            dropout=args.dropout).to(args.device)
    if args.model_type == 'GlobalAttentionNet':
        model = GlobalAttentionNet(
            in_channels=dataset.num_features,
            hidden=args.trans_hidden,
            out_channels=dataset.num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout).to(args.device)
    if args.model_type == 'ASAP':
        model = ASAP(
            in_channels=dataset.num_features,
            hidden=args.trans_hidden,
            out_channels=dataset.num_classes,
            num_layers=args.num_layers,
            pooling=args.pooling,
            dropout=args.dropout).to(args.device)
    """MLP+GNN"""
    if args.model_type == 'All_Concat_model_MLP_GCN_test':
        model = All_Concat_model_MLP_GCN_test(trans_input_size=dataset.num_features,
                                              trans_hidden=args.trans_hidden,
                                              code_input_size=dataset.num_code_attributes,
                                              code_hidden=args.code_hidden,
                                              output_size=args.channel_hidden,
                                              num_layers=args.num_layers,
                                              pooling=args.pooling,
                                              dropout=args.dropout,
                                              final_size=dataset.num_classes).to(args.device)
    if args.model_type == 'All_Concat_model_MLP_GAT_test':
        model = All_Concat_model_MLP_GAT_test(trans_input_size=dataset.num_features,
                                              trans_hidden=args.trans_hidden,
                                              code_input_size=dataset.num_code_attributes,
                                              code_hidden=args.code_hidden,
                                              output_size=args.channel_hidden,
                                              num_layers=args.num_layers,
                                              pooling=args.pooling,
                                              dropout=args.dropout,
                                              final_size=dataset.num_classes).to(args.device)
    if args.model_type == 'All_Concat_model_MLP_SAGPool_test':
        model = All_Concat_model_MLP_SAGPool_test(trans_input_size=dataset.num_features,
                                                  trans_hidden=args.trans_hidden,
                                                  code_input_size=dataset.num_code_attributes,
                                                  code_hidden=args.code_hidden,
                                                  output_size=args.channel_hidden,
                                                  num_layers=args.num_layers,
                                                  pooling=args.pooling,
                                                  dropout=args.dropout,
                                                  final_size=dataset.num_classes).to(args.device)
    if args.model_type == 'All_Concat_model_MLP_GA_test':
        model = All_Concat_model_MLP_GA_test(trans_input_size=dataset.num_features,
                                             trans_hidden=args.trans_hidden,
                                             code_input_size=dataset.num_code_attributes,
                                             code_hidden=args.code_hidden,
                                             output_size=args.channel_hidden,
                                             num_layers=args.num_layers,
                                             pooling=args.pooling,
                                             dropout=args.dropout,
                                             final_size=dataset.num_classes).to(args.device)
    if args.model_type == 'All_Concat_model_MLP_ASAP_test':
        model = All_Concat_model_MLP_ASAP_test(trans_input_size=dataset.num_features,
                                               trans_hidden=args.trans_hidden,
                                               code_input_size=dataset.num_code_attributes,
                                               code_hidden=args.code_hidden,
                                               output_size=args.channel_hidden,
                                               num_layers=args.num_layers,
                                               pooling=args.pooling,
                                               dropout=args.dropout,
                                               final_size=dataset.num_classes).to(args.device)
    """no_MLP+GNN"""
    if args.model_type == 'All_Concat_model_no_MLP_GCN_test':
        model = All_Concat_model_no_MLP_GCN_test(trans_input_size=dataset.num_features,
                                                      trans_hidden=args.trans_hidden,
                                                      code_input_size=dataset.num_code_attributes,
                                                      code_hidden=args.code_hidden,
                                                      output_size=args.channel_hidden,
                                                      num_layers=args.num_layers,
                                                      pooling=args.pooling,
                                                      dropout=args.dropout,
                                                      final_size=dataset.num_classes).to(args.device)
    if args.model_type == 'All_Concat_model_no_MLP_GAT_test':
        model = All_Concat_model_no_MLP_GAT_test(trans_input_size=dataset.num_features,
                                                      trans_hidden=args.trans_hidden,
                                                      code_input_size=dataset.num_code_attributes,
                                                      code_hidden=args.code_hidden,
                                                      output_size=args.channel_hidden,
                                                      num_layers=args.num_layers,
                                                      pooling=args.pooling,
                                                      dropout=args.dropout,
                                                      final_size=dataset.num_classes).to(args.device)
    if args.model_type == 'All_Concat_model_no_MLP_SAGPool_test':
        model = All_Concat_model_no_MLP_SAGPool_test(trans_input_size=dataset.num_features,
                                                          trans_hidden=args.trans_hidden,
                                                          code_input_size=dataset.num_code_attributes,
                                                          code_hidden=args.code_hidden,
                                                          output_size=args.channel_hidden,
                                                          num_layers=args.num_layers,
                                                          pooling=args.pooling,
                                                          dropout=args.dropout,
                                                          final_size=dataset.num_classes).to(args.device)
    if args.model_type == 'All_Concat_model_no_MLP_GA_test':
        model = All_Concat_model_no_MLP_GA_test(trans_input_size=dataset.num_features,
                                                     trans_hidden=args.trans_hidden,
                                                     code_input_size=dataset.num_code_attributes,
                                                     code_hidden=args.code_hidden,
                                                     output_size=args.channel_hidden,
                                                     num_layers=args.num_layers,
                                                     pooling=args.pooling,
                                                     dropout=args.dropout,
                                                     final_size=dataset.num_classes).to(args.device)
    if args.model_type == 'All_Concat_model_no_MLP_ASAP_test':
        model = All_Concat_model_no_MLP_ASAP_test(trans_input_size=dataset.num_features,
                                                       trans_hidden=args.trans_hidden,
                                                       code_input_size=dataset.num_code_attributes,
                                                       code_hidden=args.code_hidden,
                                                       output_size=args.channel_hidden,
                                                       num_layers=args.num_layers,
                                                       pooling=args.pooling,
                                                       dropout=args.dropout,
                                                       final_size=dataset.num_classes).to(args.device)

    """KNNG+GNN"""
    if args.model_type == 'All_Concat_model_GCN_GCN_test':
        model = All_Concat_model_GCN_GCN_test(trans_input_size=dataset.num_features,
                                                   trans_hidden=args.trans_hidden,
                                                   code_input_size=dataset.num_code_attributes,
                                                   code_hidden=args.code_hidden,
                                                   output_size=args.channel_hidden,
                                                   num_layers=args.num_layers,
                                                   pooling=args.pooling,
                                                   dropout=args.dropout,
                                                   final_size=dataset.num_classes).to(args.device)
    if args.model_type == 'All_Concat_model_GCN_GAT_test':
        model = All_Concat_model_GCN_GAT_test(trans_input_size=dataset.num_features,
                                                   trans_hidden=args.trans_hidden,
                                                   code_input_size=dataset.num_code_attributes,
                                                   code_hidden=args.code_hidden,
                                                   output_size=args.channel_hidden,
                                                   num_layers=args.num_layers,
                                                   pooling=args.pooling,
                                                   dropout=args.dropout,
                                                   final_size=dataset.num_classes).to(args.device)
    if args.model_type == 'All_Concat_model_GCN_SAGPool_test':
        model = All_Concat_model_GCN_SAGPool_test(trans_input_size=dataset.num_features,
                                                       trans_hidden=args.trans_hidden,
                                                       code_input_size=dataset.num_code_attributes,
                                                       code_hidden=args.code_hidden,
                                                       output_size=args.channel_hidden,
                                                       num_layers=args.num_layers,
                                                       pooling=args.pooling,
                                                       dropout=args.dropout,
                                                       final_size=dataset.num_classes).to(args.device)
    if args.model_type == 'All_Concat_model_GCN_GA_test':
        model = All_Concat_model_GCN_GA_test(trans_input_size=dataset.num_features,
                                                  trans_hidden=args.trans_hidden,
                                                  code_input_size=dataset.num_code_attributes,
                                                  code_hidden=args.code_hidden,
                                                  output_size=args.channel_hidden,
                                                  num_layers=args.num_layers,
                                                  pooling=args.pooling,
                                                  dropout=args.dropout,
                                                  final_size=dataset.num_classes).to(args.device)
    if args.model_type == 'All_Concat_model_GCN_ASAP_test':
        model = All_Concat_model_GCN_ASAP_test(trans_input_size=dataset.num_features,
                                                    trans_hidden=args.trans_hidden,
                                                    code_input_size=dataset.num_code_attributes,
                                                    code_hidden=args.code_hidden,
                                                    output_size=args.channel_hidden,
                                                    num_layers=args.num_layers,
                                                    pooling=args.pooling,
                                                    dropout=args.dropout,
                                                    final_size=dataset.num_classes).to(args.device)
    if args.model_type == 'All_Concat_model_GAT_GCN_test':
        model = All_Concat_model_GAT_GCN_test(trans_input_size=dataset.num_features,
                                                   trans_hidden=args.trans_hidden,
                                                   code_input_size=dataset.num_code_attributes,
                                                   code_hidden=args.code_hidden,
                                                   output_size=args.channel_hidden,
                                                   num_layers=args.num_layers,
                                                   pooling=args.pooling,
                                                   dropout=args.dropout,
                                                   final_size=dataset.num_classes).to(args.device)
    if args.model_type == 'All_Concat_model_GAT_GAT_test':
        model = All_Concat_model_GAT_GAT_test(trans_input_size=dataset.num_features,
                                                   trans_hidden=args.trans_hidden,
                                                   code_input_size=dataset.num_code_attributes,
                                                   code_hidden=args.code_hidden,
                                                   output_size=args.channel_hidden,
                                                   num_layers=args.num_layers,
                                                   pooling=args.pooling,
                                                   dropout=args.dropout,
                                                   final_size=dataset.num_classes).to(args.device)
    if args.model_type == 'All_Concat_model_GAT_SAGPool_test':
        model = All_Concat_model_GAT_SAGPool_test(trans_input_size=dataset.num_features,
                                                       trans_hidden=args.trans_hidden,
                                                       code_input_size=dataset.num_code_attributes,
                                                       code_hidden=args.code_hidden,
                                                       output_size=args.channel_hidden,
                                                       num_layers=args.num_layers,
                                                       pooling=args.pooling,
                                                       dropout=args.dropout,
                                                       final_size=dataset.num_classes).to(args.device)
    if args.model_type == 'All_Concat_model_GAT_GA_test':
        model = All_Concat_model_GAT_GA_test(trans_input_size=dataset.num_features,
                                                  trans_hidden=args.trans_hidden,
                                                  code_input_size=dataset.num_code_attributes,
                                                  code_hidden=args.code_hidden,
                                                  output_size=args.channel_hidden,
                                                  num_layers=args.num_layers,
                                                  pooling=args.pooling,
                                                  dropout=args.dropout,
                                                  final_size=dataset.num_classes).to(args.device)
    if args.model_type == 'All_Concat_model_GAT_ASAP_test':
        model = All_Concat_model_GAT_ASAP_test(trans_input_size=dataset.num_features,
                                                    trans_hidden=args.trans_hidden,
                                                    code_input_size=dataset.num_code_attributes,
                                                    code_hidden=args.code_hidden,
                                                    output_size=args.channel_hidden,
                                                    num_layers=args.num_layers,
                                                    pooling=args.pooling,
                                                    dropout=args.dropout,
                                                    final_size=dataset.num_classes).to(args.device)

    return model