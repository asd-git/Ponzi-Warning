import json

import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os


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






def load_data_json(edges_num, rnd_state, type_, subG_dataset):
    """
    单个模型，已划分好的数据集
    """
    if type_ == 'up_100':
        path = f'/public/MountData/DataDir/jj/Ponzi_EWES_new/data/{subG_dataset}/class_data_machine/'
        json_path = path + f'/{rnd_state}/TO_train_val_data.json'
        with open(json_path) as f:
            json_data = json.load(f)
            train_trans = np.array(json_data['trans_old']['train'])
            val_trans = np.array(json_data['trans_old']['val'])
            X_trans_train = train_trans[:, :-1]
            Y_trans_train = train_trans[:, -1]
            X_trans_val = val_trans[:, :-1]
            Y_trans_val = val_trans[:, -1]

            train_code = np.array(json_data['code']['train'])
            val_code = np.array(json_data['code']['val'])
            X_code_train = train_code[:, :-1]
            Y_code_train = train_code[:, -1]
            X_code_val = val_code[:, :-1]
            Y_code_val = val_code[:, -1]

        test_trans_path = path + f'/{rnd_state}/edge_num_{edges_num}/trans_test_data.csv'
        test_trans_df = pd.read_csv(test_trans_path)

        X_trans_test = test_trans_df.iloc[:, 1:-1].to_numpy()
        Y_trans_test = test_trans_df.iloc[:, -1].to_numpy()

        test_code_path = path + f'/{rnd_state}/edge_num_{edges_num}/code_test_data.csv'
        test_code_df = pd.read_csv(test_code_path)

        X_code_test = test_code_df.iloc[:, 1:-1].to_numpy()
        Y_code_test = test_code_df.iloc[:, -1].to_numpy()

    # elif type_ == 'all':
    #     path = '/public/MountData/DataDir/jj/SG_down_dataset/class_data_machine1/'
    #
    #     json_path = path + f'/{rnd_state}/machine_train_val_data.json'
    #     with open(json_path) as f:
    #         json_data = json.load(f)
    #         train = np.array(json_data['train'])
    #         val = np.array(json_data['val'])
    #         X_train = train[:, :-1]
    #         Y_train = train[:, -1]
    #         X_val = val[:, :-1]
    #         Y_val = val[:, -1]
    #
    #     test_path = path + f'/{rnd_state}/edge_num_{edges_num}/test_data.csv'
    #     test_df = pd.read_csv(test_path)
    #     X_test = test_df.iloc[:, 1:-1].to_numpy()
    #     Y_test = test_df.iloc[:, -1].to_numpy()
    #
    #     all_X = np.vstack((X_train, X_val, X_test))
    #     normal_X = np.arctan(all_X) * (2 / np.pi)
    #
    #     X_train = normal_X[:X_train.shape[0], :]
    #
    #     X_val = normal_X[X_train.shape[0]:X_train.shape[0] + X_val.shape[0], :]
    #     X_test = normal_X[X_train.shape[0] + X_val.shape[0]:, :]
    #
    # if type_ == '1-1':
    #     path = '/public/MountData/DataDir/jj/SG_down_dataset/class_data_machine_1_1/'
    #     json_path = path + f'/{rnd_state}/machine_train_val_data.json'
    #     with open(json_path) as f:
    #         json_data = json.load(f)
    #         train = np.array(json_data['train'])
    #         val = np.array(json_data['val'])
    #         X_train = train[:, :-1]
    #         Y_train = train[:, -1]
    #         X_val = val[:, :-1]
    #         Y_val = val[:, -1]
    #
    #     test_path = path + f'/{rnd_state}/edge_num_{edges_num}/test_data.csv'
    #     test_df = pd.read_csv(test_path)
    #
    #     X_test = test_df.iloc[:, 1:-1].to_numpy()
    #     Y_test = test_df.iloc[:, -1].to_numpy()

    train_trans_x = torch.tensor(X_trans_train)
    train_trans_y = torch.LongTensor(Y_trans_train)
    test_trans_x = torch.tensor(X_trans_test)
    test_trans_y = torch.LongTensor(Y_trans_test)
    val_trans_x = torch.tensor(X_trans_val)
    val_trans_y = torch.LongTensor(Y_trans_val)

    train_code_x = torch.tensor(X_code_train)
    train_code_y = torch.LongTensor(Y_code_train)
    test_code_x = torch.tensor(X_code_test)
    test_code_y = torch.LongTensor(Y_code_test)

    val_code_x = torch.tensor(X_code_val)
    val_code_y = torch.LongTensor(Y_code_val)

    train_x = [train_trans_x.float(), train_code_x.float()]
    val_x = [val_trans_x.float(), val_code_x.float()]
    test_x = [test_trans_x.float(), test_code_x.float()]

    train_y = train_trans_y
    val_y = val_trans_y
    test_y = test_trans_y

    return train_x, val_x, test_x, train_y, val_y, test_y


def load_data_json_machine(edges_num, rnd_state, type_, subG_dataset):
    """
    单个模型，已划分好的数据集
    """
    if type_ == 'up_100':
        path = f'/public/MountData/DataDir/jj/Ponzi_EWES_new/data/{subG_dataset}/class_data_machine/'

        json_path = path + f'/{rnd_state}/TO_train_val_data.json'
        with open(json_path) as f:
            json_data = json.load(f)
            train_trans = np.array(json_data['trans_old']['train'])
            val_trans = np.array(json_data['trans_old']['val'])
            X_trans_train = train_trans[:, :-1]
            Y_trans_train = train_trans[:, -1]
            X_trans_val = val_trans[:, :-1]
            Y_trans_val = val_trans[:, -1]

            train_code = np.array(json_data['code']['train'])
            val_code = np.array(json_data['code']['val'])
            X_code_train = train_code[:, :-1]
            Y_code_train = train_code[:, -1]
            X_code_val = val_code[:, :-1]
            Y_code_val = val_code[:, -1]

        test_trans_path = path + f'/{rnd_state}/edge_num_{edges_num}/trans_test_data.csv'
        test_trans_df = pd.read_csv(test_trans_path)

        X_trans_test = test_trans_df.iloc[:, 1:-1].to_numpy()
        Y_trans_test = test_trans_df.iloc[:, -1].to_numpy()

        test_code_path = path + f'/{rnd_state}/edge_num_{edges_num}/code_test_data.csv'
        test_code_df = pd.read_csv(test_code_path)

        X_code_test = test_code_df.iloc[:, 1:-1].to_numpy()
        Y_code_test = test_code_df.iloc[:, -1].to_numpy()

    train_x = [X_trans_train, X_code_train]
    val_x = [X_trans_val, X_code_val]
    test_x = [X_trans_test, X_code_test]

    train_y = Y_trans_train
    val_y = Y_trans_val
    test_y = Y_trans_test

    return train_x, val_x, test_x, train_y, val_y, test_y


def load_data_json_mlp_new(edges_num, rnd_state, type_, subG_dataset):
    """
    单个模型，已划分好的数据集
    """

    train_code_path = '/public/MountData/DataDir/jj/Ponzi_EWES/Ponzi_detect_model/data_deal/code_train_data.csv'
    train_code_df = pd.read_csv(train_code_path)
    X_code_train = train_code_df.iloc[:, :-2].to_numpy()
    Y_code_train = train_code_df.iloc[:, -1].to_numpy()

    val_code_path = '/public/MountData/DataDir/jj/Ponzi_EWES/Ponzi_detect_model/data_deal/code_val_data.csv'
    val_code_df = pd.read_csv(val_code_path)
    X_code_val = val_code_df.iloc[:, :-2].to_numpy()
    Y_code_val = val_code_df.iloc[:, -1].to_numpy()

    test_code_path = '/public/MountData/DataDir/jj/Ponzi_EWES/Ponzi_detect_model/data_deal/code_test_data.csv'
    test_code_df = pd.read_csv(test_code_path)

    X_code_test = test_code_df.iloc[:, :-2].to_numpy()
    Y_code_test = test_code_df.iloc[:, -1].to_numpy()

    train_trans_x = 0
    val_trans_x = 0
    test_trans_x = 0

    # train_y = Y_code_train
    # val_y = Y_code_val
    # test_y = Y_code_test

    train_code_x = torch.tensor(X_code_train)
    train_code_y = torch.LongTensor(Y_code_train)
    test_code_x = torch.tensor(X_code_test)
    test_code_y = torch.LongTensor(Y_code_test)

    val_code_x = torch.tensor(X_code_val)
    val_code_y = torch.LongTensor(Y_code_val)

    train_x = [train_trans_x, train_code_x.float()]
    val_x = [val_trans_x, val_code_x.float()]
    test_x = [test_trans_x, test_code_x.float()]
    train_y = train_code_y
    val_y = val_code_y
    test_y = test_code_y

    return train_x, val_x, test_x, train_y, val_y, test_y


def choice_edge_num_spilit_idx_new(edges_num, subG_dataset, rnd_state):
    # edges_num 子图规模  10，20，30
    data = {}
    np.random.seed(rnd_state)
    # Todo 特征变了这里就要改
    save_path = f'/public/MountData/DataDir/jj/Ponzi_EWES_new/data/{subG_dataset}/graph_class_data/{rnd_state}/'
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
    data['splits'] = splits
    return data


def load_data_mlp_new(edges_num, rnd_state, type_, subG_dataset):
    """
    单个模型，已划分好的数据集
    """

    data = choice_edge_num_spilit_idx_new(edges_num, subG_dataset, rnd_state)

    train_trans_x = 0
    val_trans_x = 0
    test_trans_x = 0
    code_atrributes = f'/public/MountData/DataDir/jj/Ponzi_EWES_new/data/ETH_data_test/new_T/{subG_dataset}/Ethereum_new_T_std_True_code_attributes.txt'

    df = pd.read_csv(code_atrributes, header=None)

    labels = []
    for i in range(10):
        label = [1] * 75 + [0] * 375
        labels.extend(label)
    df['label'] = labels

    code_train = df[df.index.isin(data['splits'][0]['train'])]
    code_val = df[df.index.isin(data['splits'][0]['val'])]
    code_test = df[df.index.isin(data['splits'][0]['test'])]

    X_code_train = code_train.iloc[:, :-1].to_numpy()
    X_code_val = code_val.iloc[:, :-1].to_numpy()
    X_code_test = code_test.iloc[:, :-1].to_numpy()

    Y_code_train = code_train.iloc[:, -1].tolist()
    Y_code_val = code_val.iloc[:, -1].tolist()
    Y_code_test = code_test.iloc[:, -1].tolist()

    train_code_x = torch.tensor(X_code_train)
    train_code_y = torch.LongTensor(Y_code_train)
    test_code_x = torch.tensor(X_code_test)
    test_code_y = torch.LongTensor(Y_code_test)

    val_code_x = torch.tensor(X_code_val)
    val_code_y = torch.LongTensor(Y_code_val)

    train_x = [train_trans_x, train_code_x.float()]
    val_x = [val_trans_x, val_code_x.float()]
    test_x = [test_trans_x, test_code_x.float()]
    train_y = train_code_y
    val_y = val_code_y
    test_y = test_code_y

    return train_x, val_x, test_x, train_y, val_y, test_y



def load_data_trans_machine_new(edges_num, rnd_state, type_, subG_dataset):
    """
    单个模型，已划分好的数据集
    """

    data = choice_edge_num_spilit_idx_new(edges_num, subG_dataset, rnd_state)

    train_trans_x = 0
    val_trans_x = 0
    test_trans_x = 0

    trans_atrributes = f'/public/MountData/DataDir/jj/Ponzi_EWES_new/data/{subG_dataset}/class_data_machine/subG_10-100_PUP_node.csv'
    trans_df = pd.read_csv(trans_atrributes)

    code_train = trans_df[trans_df.index.isin(data['splits'][0]['train'])]
    code_val = trans_df[trans_df.index.isin(data['splits'][0]['val'])]
    code_test = trans_df[trans_df.index.isin(data['splits'][0]['test'])]

    X_trans_train = code_train.iloc[:, 1:-1].to_numpy()
    X_trans_val = code_val.iloc[:, 1:-1].to_numpy()
    X_trans_test = code_test.iloc[:, 1:-1].to_numpy()

    Y_trans_train = code_train.iloc[:, -1].tolist()
    Y_trans_val = code_val.iloc[:, -1].tolist()
    Y_trans_test = code_test.iloc[:, -1].tolist()


    train_trans_x = torch.tensor(X_trans_train)
    train_trans_y = torch.LongTensor(Y_trans_train)

    val_trans_x = torch.tensor(X_trans_val)
    val_trans_y = torch.LongTensor(Y_trans_val)
    test_trans_x = torch.tensor(X_trans_test)
    test_trans_y = torch.LongTensor(Y_trans_test)



    code_atrributes = f'/public/MountData/DataDir/jj/Ponzi_EWES_new/data/ETH_data_test/new_T/{subG_dataset}/Ethereum_new_T_std_True_code_attributes.txt'

    df = pd.read_csv(code_atrributes, header=None)
    labels = []
    for i in range(10):
        label = [1] * 75 + [0] * 375
        labels.extend(label)
    df['label'] = labels

    code_train = df[df.index.isin(data['splits'][0]['train'])]
    code_val = df[df.index.isin(data['splits'][0]['val'])]
    code_test = df[df.index.isin(data['splits'][0]['test'])]

    X_code_train = code_train.iloc[:, :-1].to_numpy()
    X_code_val = code_val.iloc[:, :-1].to_numpy()
    X_code_test = code_test.iloc[:, :-1].to_numpy()

    Y_code_train = code_train.iloc[:, -1].tolist()
    Y_code_val = code_val.iloc[:, -1].tolist()
    Y_code_test = code_test.iloc[:, -1].tolist()

    train_code_x = torch.tensor(X_code_train)
    train_code_y = torch.LongTensor(Y_code_train)
    test_code_x = torch.tensor(X_code_test)
    test_code_y = torch.LongTensor(Y_code_test)

    val_code_x = torch.tensor(X_code_val)
    val_code_y = torch.LongTensor(Y_code_val)

    train_x = [train_trans_x, train_code_x.float()]
    val_x = [val_trans_x, val_code_x.float()]
    test_x = [test_trans_x, test_code_x.float()]
    train_y = train_code_y
    val_y = val_code_y
    test_y = test_code_y

    return train_x, val_x, test_x, train_y, val_y, test_y