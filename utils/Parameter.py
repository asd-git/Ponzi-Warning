import argparse


def Parameters():
    parser = argparse.ArgumentParser(description='Graph Convolutional Networks')
    #————————————————————————————————————————————————————————————————————————————————————————
    """
    可调参
    """
    parser.add_argument('-fr', '--file_root', type=str, default='./data_new/',help='dataset file root path')
    parser.add_argument('-tc', '--train_connect', type=str, default=None)
    parser.add_argument( '--train_Type', type=str, default='all',help='train_Type_tip')
    parser.add_argument( '--train_test', type=str, default=None,help='train or test or train and test')

    parser.add_argument('-D', '--dataset', type=str, default='new_subG_dataset',choices=['new_subG_dataset', 'subG_dataset'])
    # 在这份中一直是trans
    # parser.add_argument('-ft', '--feature_type', type=str, default='all',choices=['trans_old', 'code','all'])
    # 模型类型
    parser.add_argument('-mt', '--model_type', type=str, default='GCN',choices=['GIN', 'GCN'])
    parser.add_argument('-pl', '--pooling', type=str, default='max',choices=['max', 'mean','add'])
    # 控制比例规模
    parser.add_argument('-pt', '--proportion', type=str, default='1:n')
    # 测试集 边的规模
    parser.add_argument('-en', '--edge_num', type=int, default=10,
                        help='different graph size,mainly conctrol test data')

    parser.add_argument('-enl', '--edge_num_ls', type=list, default=list(range(10, 110, 10)),
                        help='different graph size,mainly conctrol test data')
    # 不变
    # parser.add_argument('-ob', '--orderby', type=str, default='new_T', help='{T,new_T}')
    # 调参
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='dropout rate')


   # trans_old 层
    parser.add_argument('-th', '--trans_hidden', type=str, default=0, help='trans_code')
    # 这个是code层的
    parser.add_argument('-h1', '--code_hidden', type=str, default=0, help='code hidden')
    # parser.add_argument('-h2', '--hidden2', type=str, default=32, help='code hidden2')
    # 这个是两个层的通道输出的维度

    parser.add_argument('-ch', '--channel_hidden', type=str, default=8, help='trans_code')
    # 可以有几层
    parser.add_argument('-nl', '--num_layers', type=str, default=2, help='num_layers')


    # 邻接矩阵 （方向）  非对称、对称、对称单位阵
    parser.add_argument('--adj_type', type=str, default='sym', choices=[ 'unsym','sym', 'i'])
    # 边的类型 数量边、金额边   这个在代码中好像没体现
    parser.add_argument('--edge_type', type=str, default='count_edge', choices=['count_edge', 'V_edge'])
    # 代码特征 是否采取归一化  都是True  已经先处理好了
    parser.add_argument('--Isnormalization', type=str, default='True', choices=['False', 'True'])  # code 归一化

    # 交易特征 归一化类型  不同图进行归一化、不同规模进行归一化、还是全局归一化
    parser.add_argument('--trans_Normal', type=str, default='all_size', choices=['all_size', 'edge_num', 'graph'])
    parser.add_argument('--early_stop', type=int, help='', default=1)
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('-bs', '--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--bn', action='store_true', default=True, help='use BatchNorm layer')
    parser.add_argument('-kf', '--Kfold', type=str, default=5, help='Kfold')
    # ————————————————————————————————————————————————————————————————————————————————————————
    parser.add_argument('--folds', type=int, default=-1,
                        help='number of cross-validation folds')
    parser.add_argument('--gpu', type=str, default='1', help='gpu')
    parser.add_argument('--lr_decay_steps', type=list, default=[25,35], help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('-K', '--filter_scale', type=int, default=1,
                        help='filter scale (receptive field size), must be > 0; 1 for GCN, >1 for ChebNet')
    parser.add_argument('--n_hidden', type=int, default=0,
                        help='number of hidden units in a fully connected layer after the last conv layer')
    parser.add_argument('--log_interval', type=int, default=10, help='interval (number of batches) of logging')
    parser.add_argument('-t', '--threads', type=int, default=0, help='number of threads to load data')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'])

    parser.add_argument('--seed', type=int, default=111, help='random seed')
    parser.add_argument('--shuffle_nodes', action='store_true', default=False, help='shuffle nodes for debugging')
    return parser.parse_args()