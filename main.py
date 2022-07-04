import os
import re

import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from torch_geometric.loader import DataLoader

# from code_KNN图.my_planatoid import my_planatoid
from utils.Parameter import Parameters
from utils.my_tu_dataset import My_TUDataset
from utils.utils import mkdir, kfold_split_tvt_idx, EarlyStopping, set_all_random, print_args, model_choice, \
    ColumnNormalizeFeatures



def get_title():
    name = os.path.basename(__file__)
    regex = re.compile(r'\d+')
    namenum = regex.findall(name)
    titile = "-".join(namenum)
    return titile
import warnings
warnings.filterwarnings("ignore")


def Main(args):

    def train(train_loader):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(args.device)
            optimizer.zero_grad()
            if args.train_Type == 'all' or args.train_Type == 'all_no_MLP':
                _, _, output = model(data)
            if args.train_Type == 'code' or args.train_Type == 'trans':
                output = model(data)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def a_test(loader):
        model.eval()

        total_correct, total_loss = 0, 0
        y_pred_label_list = []
        y_true_label_list = []

        for data in loader:
            data = data.to(args.device)
            if args.train_Type == 'all' or args.train_Type == 'all_no_MLP':
                _, _, out = model(data)
            if args.train_Type == 'code' or args.train_Type == 'trans':
                out = model(data)
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs

            pred = out.argmax(-1)
            y_pred_label_list.append(pred)
            y_true_label_list.append(data.y)

        preds = torch.cat(y_pred_label_list).cpu().detach().numpy()
        reals = torch.cat(y_true_label_list).cpu().detach().numpy()

        precision = precision_score(reals, preds, average='weighted')
        recall = recall_score(reals, preds, average='weighted')
        f1 = f1_score(reals, preds, average='weighted')


        return f1, precision, recall, total_loss / len(loader.dataset)

    # 设置随机种子
    rnd_states = set_all_random(args)
    # print parameters
    # print_args(args)
    print(f"{args.train_Type} 部分")

    print('Loading data...')
    dataset = My_TUDataset(root=args.file_root, name="eth")

    # 各项指标
    f1_per_time = []
    prec_per_time = []
    rec_per_time = []
    for rnd_state in rnd_states:
        print("rnd_state:{}".format(rnd_state))
        np.random.seed(rnd_state)
        # todo 训练集、验证集、测试集的索引  (有验证集)
        data = kfold_split_tvt_idx(args.edge_num_ls, args.Kfold, rnd_state)
        for kf in range(args.Kfold):
            print("kf:{}".format(kf))
            # kf当做了random，也相当于将拥有的数据都训练测试了一遍。
            train_idx = data[kf]['train_idx']
            val_idx = data[kf]['val_idx']
            test_idx = data[kf]['test_idx'][args.edge_num]
            train_eth = dataset[train_idx]
            val_eth = dataset[val_idx]
            test_eth = dataset[test_idx]
            train_loader = DataLoader(train_eth, batch_size=200,shuffle=True)
            val_loader = DataLoader(val_eth, batch_size=200,shuffle=True)
            test_loader = DataLoader(test_eth, batch_size=200)

            # 选择 模型
            model = model_choice(args,dataset)
            print('\n初始化模型')
            print(model)
            print("_______________________________")
            print("\n优化器优化")
            # optimizer = optim.Adam(model.parameters(), lr=args.lr)

            train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
            print('N trainable parameters:', np.sum([p.numel() for p in train_params]))
            optimizer = optim.Adam(train_params, lr=args.lr, weight_decay=args.wd, betas=(0.5, 0.999))
            # 设置早停机制，开始训练
            early_stopping = EarlyStopping(patience=20)
            for epoch in range(args.epochs):
                loss = train(train_loader)
                train_f1, train_precision, train_recall, _ = a_test(train_loader)
                val_f1, val_precision, val_recall, val_loss = a_test(val_loader)
                test_f1, test_pre, test_rec, _ = a_test(test_loader)

                if args.early_stop:
                    early_stopping(val_loss,
                                   results=[epoch, loss, val_loss, train_f1, val_f1, test_f1, test_pre, test_rec])
                    if early_stopping.early_stop:
                        print('\n=====final results=====')
                        _epoch, _loss, _val_loss, _train_f1, _val_f1, _test_f1, _test_pre, _test_rec = early_stopping.best_results
                        f1_per_time.append(_test_f1)
                        prec_per_time.append(_test_pre)
                        rec_per_time.append(_test_rec)
                        print(f'Exp: {kf},  Epoch: {_epoch:03d},     '
                              f'pooling : {args.pooling}      '
                              f'Train_Loss: {_loss:.4f}, Val_Loss: {_val_loss:.4f},        '
                              f'Train_f1: {_train_f1:.4f}, Val_f1: {_val_f1:.4f},        '
                              f'Test_f1: {_test_f1:.4f}\n\n')
                        save_path = mkdir(
                            f'./{get_title()}_result_{args.file_root.split("/")[1]}/model_pkl_val/{args.train_Type}/{args.model_type}_{args.num_layers}/rnd_state_{rnd_state}/dim_{args.trans_hidden}_{args.code_hidden}_{ args.channel_hidden}/kf_{kf}/')
                        torch.save(model.state_dict(),
                                   save_path + 'model_dropout_{}_{}_{}.pkl'.format(args.dropout, rnd_state,
                                                                                   args.pooling))
                        break
                    else:
                        save_path = mkdir(
                            f'./{get_title()}_result_{args.file_root.split("/")[1]}/model_pkl_val/{args.train_Type}/{args.model_type}_{args.num_layers}/rnd_state_{rnd_state}/dim_{args.trans_hidden}_{args.code_hidden}_{args.channel_hidden}/kf_{kf}/')
                        torch.save(model.state_dict(),
                                   save_path + 'model_dropout_{}_{}_{}.pkl'.format(args.dropout, rnd_state,
                                                                                   args.pooling))


                print(f'Exp: {kf},  Epoch: {epoch:03d},     '
                      f'pooling : {args.pooling}      '
                      f'Train_Loss: {loss:.4f}, Val_Loss: {val_loss:.4f},      '
                      f'Train_f1: {train_f1:.4f}, Val_f1: {val_f1:.4f},      '
                      f'Test_f1: {test_f1:.4f}')
    mean_f1 = np.mean(f1_per_time)
    std_f1 = np.std(f1_per_time)
    mean_prec = np.mean(prec_per_time)
    mean_rec = np.mean(rec_per_time)
    print("Mean f1:{:.4f}--Std f1:{:.4f}".format(mean_f1, std_f1))
    print("Mean precision:{:.4f}".format(mean_prec))
    print("Mean recall:{:.4f}".format(mean_rec))

class test_Main():
    def __init__(self, args):
        self.args = args
        self.n_folds = self.args.folds  # cross validation
        self.rnd_states = set_all_random(args)
        # print parameters
        print('Loading data...')
        dataset = My_TUDataset(root=args.file_root, name="eth")

        # 开始测试
        for rnd_state in self.rnd_states:
            print("rnd_state:{}".format(rnd_state))
            f1_per_time = []
            prec_per_time = []
            rec_per_time = []

            np.random.seed(rnd_state)
            # todo 训练集、验证集、测试集的索引  (有验证集)
            data = kfold_split_tvt_idx(self.args.edge_num_ls, self.args.Kfold, rnd_state)
            for kf in range(self.args.Kfold):
                print("kf:{}".format(kf))
                if args.file_root.split("/")[1] == 'data_no_Aug':
                    self.test_idx = data[kf]['test_idx'][10]
                else:
                    self.test_idx = data[kf]['test_idx'][self.args.edge_num]
                test_eth = dataset[self.test_idx]
                self.test_loader = DataLoader(test_eth, batch_size=80)
                print("_______________________________")
                # 选择 模型
                self.model = model_choice(self.args, dataset)

                print('\nInitialized model')
                print(self.model)
                precision, recall, f1 = self.compute_test(rnd_state,kf,self.test_loader)
                f1_per_time.append(f1)
                prec_per_time.append(precision)
                rec_per_time.append(recall)

            mean_f1 = np.mean(f1_per_time)
            std_f1 = np.std(f1_per_time)
            mean_prec = np.mean(prec_per_time)
            mean_rec = np.mean(rec_per_time)
            print("边规模：{}".format(self.args.edge_num))
            print("Mean f1:{}".format(mean_f1))
            print("Std f1:{}".format(std_f1))
            print("Mean precision:{}".format(mean_prec))
            print("Mean recall:{}".format(mean_rec))

            savepath = f'./{get_title()}_result_{args.file_root.split("/")[1]}/gnn_val/{self.args.train_Type}/{self.args.model_type}_{args.num_layers}/new_rnd_state_{rnd_state}'
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            with open(savepath + f"/{self.args.train_Type}_result_weighted.txt", 'a')as w:
                w.writelines(
                    "trans_hidden:{}\tcode_hidden:{}\tchannel_hidden:{}\tdropout:{}\tPooling:{}\tedge_num:{}\tMean pre\tMean recall\tMean f1\tStd f1:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
                        self.args.trans_hidden, self.args.code_hidden,args.channel_hidden,self.args.dropout, self.args.pooling, self.args.edge_num,
                        mean_prec, mean_rec, mean_f1, std_f1))
    def compute_test(self, rnd_state, kf, test_loader):
        # Restore best model
        print('图规模', self.args.edge_num)
        save_path = mkdir(
            f'./{get_title()}_result_{args.file_root.split("/")[1]}/model_pkl_val/{self.args.train_Type}/{self.args.model_type}_{args.num_layers}/rnd_state_{rnd_state}/dim_{args.trans_hidden}_{args.code_hidden}_{args.channel_hidden}/kf_{kf}/')
        self.model.load_state_dict(torch.load(
            save_path + 'model_dropout_{}_{}_{}.pkl'.format(self.args.dropout, rnd_state, self.args.pooling)))
        self.model.eval()
        preds = []
        reals = []
        for data in test_loader:
            data = data.to(self.args.device)
            if args.train_Type == 'all' or args.train_Type =='all_no_MLP':
                _, _, output = self.model(data)
            if args.train_Type == 'code' or args.train_Type == 'trans':
                output = self.model(data)
            pred = torch.argmax(output, dim=1).detach().cpu()
            real = data.y.detach().cpu().view_as(pred)
            preds.append(pred)
            reals.append(real)

        preds = np.hstack(preds)
        reals = np.hstack(reals)
        f1 = f1_score(reals, preds, average='weighted')
        precision = precision_score(reals, preds, average='weighted')
        recall = recall_score(reals, preds, average='weighted')
        return precision, recall, f1

def choice_train_Type(args,train_or_test):
    # KNNG 的超参  目前看来topk8 效果最好
    topk = 8
    # 隐藏层参数
    hidden_ls = [32]
    # hidden_ls = [16, 32, 64, 128]
    """"""
    # 通道输出参数
    channel_hidden_ls = [32]
    # channel_hidden_ls = [8,16,32,64,128,256]
    """"""
    # 池化策略
    pooling_ls = ['max']
    # pooling_ls = ['max','mean','add']


    """code"""
    code_modeltype_ls = ['MLP']
    #############################################################

    """trans"""
    # trans_modeltype_ls = ['GCN_Net', 'SAGPool', 'GAT', 'GlobalAttentionNet', 'ASAP']
    trans_modeltype_ls = ['GCN_Net', 'SAGPool', 'GAT', 'GlobalAttentionNet', 'ASAP']
    # trans_modeltype_ls = ['GCN_Net',  'GAT', 'GlobalAttentionNet', 'ASAP']
    """"""
    # trans_modeltype_ls = ['GCN_Net']
    #############################################################
    """MLP+GNN"""
    all_modeltype_ls = ['All_Concat_model_MLP_GCN_test','All_Concat_model_MLP_GAT_test','All_Concat_model_MLP_SAGPool_test','All_Concat_model_MLP_GA_test']
    # all_modeltype_ls = ['All_Concat_model_MLP_GCN_test','All_Concat_model_MLP_SAGPool_test','All_Concat_model_MLP_GA_test']
    # all_modeltype_ls = ['All_Concat_model_MLP_GA_test']
    # all_modeltype_ls = ['All_Concat_model_MLP_GCN_test']
    #############################################################
    """no_MLP+GNN"""
    # all_no_MLP_modeltype_ls = ['All_Concat_model_no_MLP_SAGPool_test']
    all_no_MLP_modeltype_ls = ['All_Concat_model_no_MLP_GAT_test','All_Concat_model_no_MLP_SAGPool_test',
                        'All_Concat_model_no_MLP_ASAP_test','All_Concat_model_no_MLP_GA_test']

    """KNNG+GNN"""

    all_KNN_modeltype_ls = ['All_Concat_model_GCN_GCN_test']
    # all_KNN_modeltype_ls = ['All_Concat_model_GCN_GCN_test', 'All_Concat_model_GCN_GAT_test',
    #                     'All_Concat_model_GCN_SAGPool_test', 'All_Concat_model_GCN_GA_test',
    #                     'All_Concat_model_GCN_ASAP_test',
    #                     'All_Concat_model_GAT_GCN_test', 'All_Concat_model_GAT_GAT_test',
    #                     'All_Concat_model_GAT_SAGPool_test', 'All_Concat_model_GAT_GA_test',
    #                     'All_Concat_model_GAT_ASAP_test']
    # all_KNN_modeltype_ls = ['All_Concat_model_GCN_GCN_test', 'All_Concat_model_GCN_GAT_test',
    #                         'All_Concat_model_GCN_SAGPool_test', 'All_Concat_model_GCN_GA_test',
    #                         'All_Concat_model_GCN_ASAP_test',]
    #############################################################


    if train_or_test == 'train':
        "code"  # 没有pooling 没有dropout
        if args.train_Type =='code':
            for modeltype in code_modeltype_ls:
                args.model_type = modeltype
                args.feature_type = train_Type
                for code_dim in hidden_ls:
                    args.code_hidden = code_dim
                    Main(args)

        "trans"
        if args.train_Type == 'trans' or args.train_Type == 'concat':
            for modeltype in trans_modeltype_ls:
                args.model_type = modeltype
                for pool in pooling_ls:
                    for channel_hidden in channel_hidden_ls:
                        args.channel_hidden = channel_hidden
                        args.pooling = pool
                        for trans_dim in hidden_ls:
                            args.trans_hidden = trans_dim
                            args.feature_type = train_Type
                            Main(args)

        "MLP+GNN"
        if args.train_Type == 'all':
            """
            code 层只用了MLP
            """
            for modeltype in all_modeltype_ls:
                args.model_type = modeltype
                for pool in pooling_ls:
                    for channel_hidden in channel_hidden_ls:
                        args.channel_hidden = channel_hidden
                        args.pooling = pool
                        for trans_dim in hidden_ls:
                            for code_dim in hidden_ls:
                                args.trans_hidden = trans_dim
                                args.code_hidden = code_dim
                                args.feature_type = train_Type
                                Main(args)
        "no_MLP+GNN"
        if args.train_Type == 'all_no_MLP':
            for modeltype in all_no_MLP_modeltype_ls:
                args.model_type = modeltype
                for pool in pooling_ls:
                    for channel_hidden in channel_hidden_ls:
                        args.channel_hidden = channel_hidden
                        args.pooling = pool
                        for trans_dim in hidden_ls:
                            args.trans_hidden = trans_dim
                            args.feature_type = train_Type
                            Main(args)



    if train_or_test == 'test':
        "code"  # 没有pooling 没有dropout
        if args.train_Type == 'code':
            for modeltype in code_modeltype_ls:
                args.model_type = modeltype
                for dim in hidden_ls:
                    args.code_hidden = dim
                    test_Main(args)

        "trans"
        if args.train_Type == 'trans' or args.train_Type == 'concat':
            for modeltype in trans_modeltype_ls:
                args.model_type = modeltype
                for pool in pooling_ls:
                    for channel_hidden in channel_hidden_ls:
                        args.channel_hidden = channel_hidden
                        args.pooling = pool
                        for trans_dim in hidden_ls:
                            for edge_num in args.edge_num_ls:
                                if args.file_root.split("/")[1] == 'data_no_Aug':
                                    args.edge_num = 100
                                else:
                                    args.edge_num = edge_num
                                args.trans_hidden = trans_dim
                                test_Main(args)

        "all"
        if args.train_Type == 'all':
            for modeltype in all_modeltype_ls:
                args.model_type = modeltype
                for pool in pooling_ls:
                    for channel_hidden in channel_hidden_ls:
                        args.channel_hidden = channel_hidden
                        args.pooling = pool
                        for trans_dim in hidden_ls:
                            for code_dim in hidden_ls:
                                for edge_num in args.edge_num_ls:
                                    if args.file_root.split("/")[1] == 'data_no_Aug':
                                        args.edge_num = 100
                                    else:
                                        args.edge_num = edge_num
                                    args.trans_hidden = trans_dim
                                    args.code_hidden = code_dim
                                    test_Main(args)

        'all_no_MLP'
        if args.train_Type == 'all_no_MLP':
            for modeltype in all_no_MLP_modeltype_ls:
                args.model_type = modeltype
                for pool in pooling_ls:
                    for channel_hidden in channel_hidden_ls:
                        args.channel_hidden = channel_hidden
                        args.pooling = pool
                        for trans_dim in hidden_ls:
                            for edge_num in args.edge_num_ls:
                                if args.file_root.split("/")[1] == 'data_no_Aug':
                                    args.edge_num = 100
                                else:
                                    args.edge_num = edge_num
                                args.trans_hidden = trans_dim
                                test_Main(args)




if __name__ == '__main__':
    train_Type = ['code', 'trans', 'all', 'all_no_MLP']
    args = Parameters()
####################################
    # "数据集选择  (扩充 不扩充)"
    # dataset = 'no aug'
    dataset = 'aug'
####################################
    if dataset == 'aug':
        args.file_root = './data_new'
    if dataset == 'no aug':
        args.edge_num_ls = list(range(10, 20, 10))
        args.file_root = './data_no_aug'
    if dataset == 'concat':
        args.file_root = './data_code_trans_concat'

    # 变量 层数
    args.num_layers = 2
#######################################
    args.train_connect = 'all'
    args.train_test = 'train_and_test'
#######################################

    if args.train_connect == 'code':
        print("code 部分")
        args.train_Type = train_Type[0]
        if args.train_test == 'train':
            choice_train_Type(args,'train')
        if args.train_test == 'test':
            choice_train_Type(args,'test')
        if args.train_test == 'train_and_test':
            choice_train_Type(args, 'train')
            choice_train_Type(args, 'test')
    if args.train_connect == "trans":
        print("only trans 部分")
        args.train_Type = train_Type[1]
        if args.train_test == 'train':
            choice_train_Type(args, 'train')
        if args.train_test == 'test':
            choice_train_Type(args, 'test')
        if args.train_test == 'train_and_test':
            choice_train_Type(args, 'train')
            choice_train_Type(args, 'test')
    if args.train_connect == "all":
        print("MLP+GNN 部分")
        args.train_Type = train_Type[2]
        if args.train_test == 'train':
            choice_train_Type(args, 'train')
        if args.train_test == 'test':
            choice_train_Type(args, 'test')
        if args.train_test == 'train_and_test':
            choice_train_Type(args, 'train')
            choice_train_Type(args, 'test')
    if args.train_connect == "all_no_MLP":
        print("no_MLP+GNN 部分")
        args.train_Type = train_Type[3]
        if args.train_test == 'train':
            choice_train_Type(args, 'train')
        if args.train_test == 'test':
            choice_train_Type(args, 'test')
        if args.train_test == 'train_and_test':
            choice_train_Type(args, 'train')
            choice_train_Type(args, 'test')


    # 直接拼接效果极差
    if args.train_connect == "concat":
        print("code trans concat 部分")
        args.train_Type = train_Type[1]
        if args.train_test == 'train':
            choice_train_Type(args, 'train')
        if args.train_test == 'test':
            choice_train_Type(args, 'test')
        if args.train_test == 'train_and_test':
            choice_train_Type(args, 'train')
            choice_train_Type(args, 'test')
