from model.Graph_classication.Global_attention import GlobalAttentionNet
from model.Graph_classication.asap import ASAP
from model.Graph_classication.sag_pool import SAGPool
from model.models import GCN_Net, AutoEncoder, MLP, GAT_Net, GCN_Net_1, GAT_Net_1
import torch.nn.functional as F
from layers import *

class All_Concat_model_MLP_GCN_test(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size,num_layers,pooling,dropout, final_size):
        super(All_Concat_model_MLP_GCN_test, self).__init__()
        self.GCN = GCN_Net(trans_input_size, trans_hidden, output_size,num_layers,pooling,dropout)
        self.MLP = MLP(code_input_size, code_hidden, output_size)
        self.linear_one = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(output_size + output_size, final_size)

    def forward(self, data):
        code_emb = self.MLP(data)
        trans_emb = self.GCN(data)
        code_prob = F.log_softmax(self.linear_one(code_emb),dim=-1)
        trans_prob =  F.log_softmax(self.linear_one(trans_emb),dim=-1)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb, trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)

        return code_prob,trans_prob,final_prob


class All_Concat_model_MLP_GAT_test(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size,num_layers,pooling,dropout, final_size):
        super(All_Concat_model_MLP_GAT_test, self).__init__()
        self.GNN = GAT_Net(trans_input_size, trans_hidden, output_size,num_layers,pooling,dropout)
        self.code_layer = MLP(code_input_size,code_hidden, output_size)
        self.linear_one = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(output_size + output_size, final_size)

    def forward(self, data):
        code_emb = self.code_layer(data)
        trans_emb = self.GNN(data)
        code_prob = F.log_softmax(self.linear_one(code_emb), dim=-1)
        trans_prob = F.log_softmax(self.linear_one(trans_emb), dim=-1)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb, trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)

        return code_prob,trans_prob,final_prob

class All_Concat_model_MLP_SAGPool_test(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size,num_layers,pooling,dropout, final_size):
        super(All_Concat_model_MLP_SAGPool_test, self).__init__()
        self.GNN = SAGPool(trans_input_size, trans_hidden, output_size,num_layers,pooling,dropout)
        self.code_layer = MLP(code_input_size,code_hidden, output_size)
        self.linear_one = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(output_size + output_size, final_size)

    def forward(self, data):
        code_emb = self.code_layer(data)
        trans_emb = self.GNN(data)
        code_prob = F.log_softmax(self.linear_one(code_emb), dim=-1)
        trans_prob = F.log_softmax(self.linear_one(trans_emb), dim=-1)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb, trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)

        return code_prob,trans_prob,final_prob

class All_Concat_model_MLP_GA_test(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size,num_layers,pooling,dropout, final_size):
        super(All_Concat_model_MLP_GA_test, self).__init__()
        self.GNN = GlobalAttentionNet(trans_input_size, trans_hidden, output_size,num_layers,dropout)
        self.code_layer = MLP(code_input_size,code_hidden, output_size)
        self.linear_one = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(output_size + output_size, final_size)

    def forward(self, data):
        code_emb = self.code_layer(data)
        trans_emb = self.GNN(data)
        code_prob = F.log_softmax(self.linear_one(code_emb), dim=-1)
        trans_prob = F.log_softmax(self.linear_one(trans_emb), dim=-1)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb, trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)

        return code_prob,trans_prob,final_prob

class All_Concat_model_MLP_ASAP_test(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size,num_layers,pooling,dropout, final_size):
        super(All_Concat_model_MLP_ASAP_test, self).__init__()
        self.GNN = ASAP(trans_input_size, trans_hidden, output_size,num_layers,pooling,dropout)
        self.code_layer = MLP(code_input_size,code_hidden, output_size)
        self.linear_one = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(output_size + output_size, final_size)

    def forward(self, data):
        code_emb = self.code_layer(data)
        trans_emb = self.GNN(data)
        code_prob = F.log_softmax(self.linear_one(code_emb), dim=-1)
        trans_prob = F.log_softmax(self.linear_one(trans_emb), dim=-1)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb, trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)



        return code_prob,trans_prob,final_prob


class All_Concat_model_no_MLP_GCN_test(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size,num_layers,pooling,dropout, final_size):
        super(All_Concat_model_no_MLP_GCN_test, self).__init__()
        self.GCN = GCN_Net(trans_input_size, trans_hidden, output_size,num_layers,pooling,dropout)
        # self.Linear_code = MLP(code_input_size, code_hidden, output_size)
        self.linear_code = nn.Linear(code_input_size, final_size)
        self.linear_trans = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(code_input_size + output_size, final_size)

    def forward(self, data):
        code_emb = data.code_x
        trans_emb = self.GCN(data)
        code_prob = F.log_softmax(self.linear_code(code_emb),dim=-1)
        trans_prob = F.log_softmax(self.linear_trans(trans_emb),dim=-1)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb, trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)

        return code_prob,trans_prob,final_prob


class All_Concat_model_no_MLP_GAT_test(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size,num_layers,pooling,dropout, final_size):
        super(All_Concat_model_no_MLP_GAT_test, self).__init__()
        self.GNN = GAT_Net(trans_input_size, trans_hidden, output_size,num_layers,pooling,dropout)
        # self.code_layer = MLP(code_input_size,code_hidden, output_size)
        self.linear_code = nn.Linear(code_input_size, final_size)
        self.linear_trans = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(code_input_size + output_size, final_size)
    def forward(self, data):
        code_emb = data.code_x
        trans_emb = self.GNN(data)
        code_prob = F.log_softmax(self.linear_code(code_emb), dim=-1)
        trans_prob = F.log_softmax(self.linear_trans(trans_emb), dim=-1)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb, trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)

        return code_prob,trans_prob,final_prob


class All_Concat_model_no_MLP_SAGPool_test(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size,num_layers,pooling,dropout, final_size):
        super(All_Concat_model_no_MLP_SAGPool_test, self).__init__()
        self.GNN = SAGPool(trans_input_size, trans_hidden, output_size,num_layers,pooling,dropout)
        # self.code_layer = MLP(code_input_size,code_hidden, output_size)
        self.linear_code = nn.Linear(code_input_size, final_size)
        self.linear_trans = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(code_input_size + output_size, final_size)
    def forward(self, data):
        code_emb = data.code_x
        trans_emb = self.GNN(data)
        code_prob = F.log_softmax(self.linear_code(code_emb), dim=-1)
        trans_prob = F.log_softmax(self.linear_trans(trans_emb), dim=-1)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb, trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)

        return code_prob,trans_prob,final_prob

class All_Concat_model_no_MLP_GA_test(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size,num_layers,pooling,dropout, final_size):
        super(All_Concat_model_no_MLP_GA_test, self).__init__()
        self.GNN = GlobalAttentionNet(trans_input_size, trans_hidden, output_size,num_layers,dropout)
        # self.code_layer = MLP(code_input_size,code_hidden, output_size)
        self.linear_code = nn.Linear(code_input_size, final_size)
        self.linear_trans = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(code_input_size + output_size, final_size)
    def forward(self, data):
        code_emb = data.code_x
        trans_emb = self.GNN(data)
        code_prob = F.log_softmax(self.linear_code(code_emb), dim=-1)
        trans_prob = F.log_softmax(self.linear_trans(trans_emb), dim=-1)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb, trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)

        return code_prob,trans_prob,final_prob

class All_Concat_model_no_MLP_ASAP_test(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size,num_layers,pooling,dropout, final_size):
        super(All_Concat_model_no_MLP_ASAP_test, self).__init__()
        self.GNN = ASAP(trans_input_size, trans_hidden, output_size,num_layers,pooling,dropout)
        # self.code_layer = MLP(code_input_size,code_hidden, output_size)
        self.linear_code = nn.Linear(code_input_size, final_size)
        self.linear_trans = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(code_input_size + output_size, final_size)
    def forward(self, data):
        code_emb = data.code_x
        trans_emb = self.GNN(data)
        code_prob = F.log_softmax(self.linear_code(code_emb), dim=-1)
        trans_prob = F.log_softmax(self.linear_trans(trans_emb), dim=-1)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb, trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)



        return code_prob,trans_prob,final_prob



####################################################
# 代码部分用的是GCN
class All_Concat_model_GCN_GCN_test(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size,num_layers,pooling,dropout, final_size):
        super(All_Concat_model_GCN_GCN_test, self).__init__()
        self.GCN = GCN_Net(trans_input_size, trans_hidden, output_size,num_layers,pooling,dropout)
        # self.MLP = MLP(code_input_size, code_hidden, output_size)
        self.code_GNN = GCN_Net_1(code_input_size, code_hidden, output_size)
        self.linear_one = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(output_size + output_size, final_size)


    def forward(self, data,code_data):
        if int(data.batch[-1]+1) == 256:
            mask = code_data.train_mask
        if int(data.batch[-1]+1) == 64:
            mask = code_data.val_mask
        if int(data.batch[-1]+1) == 80:
            mask = code_data.test_mask

        code_emb = self.code_GNN(code_data)
        trans_emb = self.GCN(data)
        code_prob = F.log_softmax(self.linear_one(code_emb[mask]),dim=-1)
        trans_prob =  F.log_softmax(self.linear_one(trans_emb),dim=-1)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb[mask], trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)

        return code_prob,trans_prob,final_prob

class All_Concat_model_GCN_GAT_test(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size,num_layers,pooling,dropout, final_size):
        super(All_Concat_model_GCN_GAT_test, self).__init__()
        self.GCN = GAT_Net(trans_input_size, trans_hidden, output_size,num_layers,pooling,dropout)
        # self.MLP = MLP(code_input_size, code_hidden, output_size)
        self.code_GNN = GCN_Net_1(code_input_size, code_hidden, output_size)
        self.linear_one = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(output_size + output_size, final_size)

    def forward(self, data,code_data):
        if int(data.batch[-1]+1) == 256:
            mask = code_data.train_mask
        if int(data.batch[-1]+1) == 64:
            mask = code_data.val_mask
        if int(data.batch[-1]+1) == 80:
            mask = code_data.test_mask

        code_emb = self.code_GNN(code_data)
        trans_emb = self.GCN(data)
        code_prob = F.log_softmax(self.linear_one(code_emb[mask]),dim=-1)
        trans_prob =  F.log_softmax(self.linear_one(trans_emb),dim=-1)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb[mask], trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)

        return code_prob,trans_prob,final_prob

class All_Concat_model_GCN_SAGPool_test(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size,num_layers,pooling,dropout, final_size):
        super(All_Concat_model_GCN_SAGPool_test, self).__init__()
        self.GNN = SAGPool(trans_input_size, trans_hidden, output_size,num_layers,pooling,dropout)

        self.code_GNN = GCN_Net_1(code_input_size, code_hidden, output_size)
        self.linear_one = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(output_size + output_size, final_size)

    def forward(self, data,code_data):
        if int(data.batch[-1]+1) == 256:
            mask = code_data.train_mask
        if int(data.batch[-1]+1) == 64:
            mask = code_data.val_mask
        if int(data.batch[-1]+1) == 80:
            mask = code_data.test_mask

        code_emb = self.code_GNN(code_data)
        trans_emb = self.GNN(data)
        code_prob = F.log_softmax(self.linear_one(code_emb[mask]),dim=-1)
        trans_prob =  F.log_softmax(self.linear_one(trans_emb),dim=-1)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb[mask], trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)

        return code_prob,trans_prob,final_prob

class All_Concat_model_GCN_GA_test(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size, num_layers, pooling,
                 dropout, final_size):
        super(All_Concat_model_GCN_GA_test, self).__init__()
        self.GNN = GlobalAttentionNet(trans_input_size, trans_hidden, output_size,num_layers,dropout)

        self.code_GNN = GCN_Net_1(code_input_size, code_hidden, output_size)
        self.linear_one = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(output_size + output_size, final_size)

    def forward(self, data, code_data):
        if int(data.batch[-1] + 1) == 256:
            mask = code_data.train_mask
        if int(data.batch[-1] + 1) == 64:
            mask = code_data.val_mask
        if int(data.batch[-1] + 1) == 80:
            mask = code_data.test_mask

        code_emb = self.code_GNN(code_data)
        trans_emb = self.GNN(data)
        code_prob = F.log_softmax(self.linear_one(code_emb[mask]), dim=-1)
        trans_prob = F.log_softmax(self.linear_one(trans_emb), dim=-1)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb[mask], trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)

        return code_prob, trans_prob, final_prob

class All_Concat_model_GCN_ASAP_test(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size, num_layers, pooling,
                 dropout, final_size):
        super(All_Concat_model_GCN_ASAP_test, self).__init__()
        self.GNN = ASAP(trans_input_size, trans_hidden, output_size,num_layers,pooling,dropout)

        self.code_GNN = GCN_Net_1(code_input_size, code_hidden, output_size)
        self.linear_one = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(output_size + output_size, final_size)


    def forward(self, data, code_data):
        if int(data.batch[-1] + 1) == 256:
            mask = code_data.train_mask
        if int(data.batch[-1] + 1) == 64:
            mask = code_data.val_mask
        if int(data.batch[-1] + 1) == 80:
            mask = code_data.test_mask

        code_emb = self.code_GNN(code_data)
        trans_emb = self.GNN(data)
        code_prob = F.log_softmax(self.linear_one(code_emb[mask]), dim=-1)
        trans_prob = F.log_softmax(self.linear_one(trans_emb), dim=-1)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb[mask], trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)

        return code_prob, trans_prob, final_prob

# 后者是GAT
class All_Concat_model_GAT_GCN_test(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size,num_layers,pooling,dropout, final_size):
        super(All_Concat_model_GAT_GCN_test, self).__init__()
        self.GNN = GCN_Net(trans_input_size, trans_hidden, output_size,num_layers,pooling,dropout)
        # self.MLP = MLP(code_input_size, code_hidden, output_size)
        self.code_GNN = GAT_Net_1(code_input_size, code_hidden, output_size)
        self.linear_one = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(output_size + output_size, final_size)


    def forward(self, data,code_data):
        if int(data.batch[-1]+1) == 256:
            mask = code_data.train_mask
        if int(data.batch[-1]+1) == 64:
            mask = code_data.val_mask
        if int(data.batch[-1]+1) == 80:
            mask = code_data.test_mask

        code_emb = self.code_GNN(code_data)
        trans_emb = self.GNN(data)
        code_prob = F.log_softmax(self.linear_one(code_emb[mask]),dim=-1)
        trans_prob =  F.log_softmax(self.linear_one(trans_emb),dim=-1)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb[mask], trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)

        return code_prob,trans_prob,final_prob

class All_Concat_model_GAT_GAT_test(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size, num_layers, pooling,
                 dropout, final_size):
        super(All_Concat_model_GAT_GAT_test, self).__init__()
        self.GNN = GAT_Net(trans_input_size, trans_hidden, output_size, num_layers, pooling, dropout)
        # self.MLP = MLP(code_input_size, code_hidden, output_size)
        self.code_GNN = GAT_Net_1(code_input_size, code_hidden, output_size)
        self.linear_one = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(output_size + output_size, final_size)

    def forward(self, data, code_data):
        if int(data.batch[-1] + 1) == 256:
            mask = code_data.train_mask
        if int(data.batch[-1] + 1) == 64:
            mask = code_data.val_mask
        if int(data.batch[-1] + 1) == 80:
            mask = code_data.test_mask

        code_emb = self.code_GNN(code_data)
        trans_emb = self.GNN(data)
        code_prob = F.log_softmax(self.linear_one(code_emb[mask]), dim=-1)
        trans_prob = F.log_softmax(self.linear_one(trans_emb), dim=-1)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb[mask], trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)

        return code_prob, trans_prob, final_prob

class All_Concat_model_GAT_SAGPool_test(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size, num_layers, pooling,
                 dropout, final_size):
        super(All_Concat_model_GAT_SAGPool_test, self).__init__()
        self.GNN = SAGPool(trans_input_size, trans_hidden, output_size, num_layers, pooling, dropout)

        self.code_GNN = GAT_Net_1(code_input_size, code_hidden, output_size)
        self.linear_one = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(output_size + output_size, final_size)


    def forward(self, data, code_data):
        if int(data.batch[-1] + 1) == 256:
            mask = code_data.train_mask
        if int(data.batch[-1] + 1) == 64:
            mask = code_data.val_mask
        if int(data.batch[-1] + 1) == 80:
            mask = code_data.test_mask

        code_emb = self.code_GNN(code_data)
        trans_emb = self.GNN(data)
        code_prob = F.log_softmax(self.linear_one(code_emb[mask]), dim=-1)
        trans_prob = F.log_softmax(self.linear_one(trans_emb), dim=-1)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb[mask], trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)

        return code_prob, trans_prob, final_prob

class All_Concat_model_GAT_GA_test(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size, num_layers, pooling,
                 dropout, final_size):
        super(All_Concat_model_GAT_GA_test, self).__init__()
        self.GNN = GlobalAttentionNet(trans_input_size, trans_hidden, output_size, num_layers, dropout)

        self.code_GNN = GAT_Net_1(code_input_size, code_hidden, output_size)
        self.linear_one = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(output_size + output_size, final_size)


    def forward(self, data, code_data):
        if int(data.batch[-1] + 1) == 256:
            mask = code_data.train_mask
        if int(data.batch[-1] + 1) == 64:
            mask = code_data.val_mask
        if int(data.batch[-1] + 1) == 80:
            mask = code_data.test_mask

        code_emb = self.code_GNN(code_data)
        trans_emb = self.GNN(data)
        code_prob = F.log_softmax(self.linear_one(code_emb[mask]), dim=-1)
        trans_prob = F.log_softmax(self.linear_one(trans_emb), dim=-1)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb[mask], trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)

        return code_prob, trans_prob, final_prob

class All_Concat_model_GAT_ASAP_test(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size, num_layers, pooling,
                 dropout, final_size):
        super(All_Concat_model_GAT_ASAP_test, self).__init__()
        self.GNN = ASAP(trans_input_size, trans_hidden, output_size, num_layers, pooling, dropout)

        self.code_GNN = GAT_Net_1(code_input_size, code_hidden, output_size)
        self.linear_one = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(output_size + output_size, final_size)

    def forward(self, data, code_data):
        if int(data.batch[-1] + 1) == 256:
            mask = code_data.train_mask
        if int(data.batch[-1] + 1) == 64:
            mask = code_data.val_mask
        if int(data.batch[-1] + 1) == 80:
            mask = code_data.test_mask

        code_emb = self.code_GNN(code_data)
        trans_emb = self.GNN(data)
        code_prob = F.log_softmax(self.linear_one(code_emb[mask]), dim=-1)
        trans_prob = F.log_softmax(self.linear_one(trans_emb), dim=-1)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb[mask], trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)

        return code_prob, trans_prob, final_prob