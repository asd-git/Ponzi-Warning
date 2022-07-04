import math
import os
import sys
import torch.nn.functional as F
import torch
# from torch_geometric.graphgym import GCNConv
from torch_geometric.nn import GCNConv,GATConv,GATv2Conv,JumpingKnowledge
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool, global_mean_pool
import torch_geometric.transforms as T
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, BatchNorm2d
from torch_geometric.nn import GIN, MLP, global_add_pool
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from layers import *
pooling_dict = {'add': global_add_pool,
                'mean': global_mean_pool,
                'max': global_max_pool}


class GIN_Net2(torch.nn.Module):
    """
    add
    """

    def __init__(self, in_channels, dim, out_channels):
        super().__init__()

        self.conv1 = GINConv(
            Sequential(Linear(in_channels, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


class GCN_Net(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels, num_layers,pooling,dropout=0.2):
        super().__init__()
        self.pooling = pooling
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, dim)
        # GCNConv
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(dim, dim))
        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index,edge_weight=None))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index,edge_weight=None))
        x = pooling_dict[self.pooling](x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

# todo
class GCN_Net_jump(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels, num_layers,pooling,dropout=0.2):
        super().__init__()
        self.pooling = pooling
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, dim)
        # GCNConv
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(dim, dim))
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(2*dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [pooling_dict[self.pooling](x, batch)]
        for i, conv in  enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            xs += [pooling_dict[self.pooling](x, batch)]
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


class GCNWithJK(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, mode='cat'):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]
        x = self.jump(xs)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

class GCN1_Net(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels,pooling):
        super().__init__()
        self.pooling = pooling
        self.conv1 = GCNConv(in_channels, dim)
        self.lin1 = Linear(dim, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, edge_index,edge_weight=None))
        x = pooling_dict[self.pooling](x, batch)
        x = F.elu(self.lin1(x))
        return F.log_softmax(x, dim=-1)

class GCN_Net_attr(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels, num_layers,pooling,dropout=0.2):
        super().__init__()
        self.pooling = pooling
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, dim)
        # GCNConv
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(dim+2, dim))
        self.lin1 = Linear(dim+2, dim)
        self.lin2 = Linear(dim, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch,edge_attr = data.x, data.edge_index, data.batch,data.edge_attr
        x = F.relu(self.conv1(x, edge_index,edge_attr))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index,edge_attr))
        x = pooling_dict[self.pooling](x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)



class GAT_Net(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels, num_layers,pooling,dropout=0.2):
        super().__init__()
        self.pooling = pooling
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, dim)
        # GCNConv
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GATConv(dim, dim))
        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = pooling_dict[self.pooling](x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

class GATv2_Net(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels, num_layers,pooling,dropout=0.2):
        super().__init__()
        self.pooling = pooling
        self.dropout = dropout
        self.conv1 = GATv2Conv(in_channels, dim)
        # GCNConv
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GATv2Conv(dim, dim))
        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = pooling_dict[self.pooling](x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

class GCN_2_max(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels, num_layers):
        super().__init__()
        self.conv1 = GCNConv(in_channels, dim,)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(dim, dim))
        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_max_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


class GCN4GC_1(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels, num_layers, pooling, BN=True, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        self.BN = BN
        self.dropout = dropout
        self.gcs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.drops = torch.nn.ModuleList()
        for i in range(num_layers):
            if i:
                gc = GCNConv(dim, dim)
            else:
                gc = GCNConv(in_channels, dim)
            bn = BatchNorm1d(dim)
            drop = nn.Dropout(p=dropout)
            self.gcs.append(gc)
            self.bns.append(bn)
            self.drops.append(drop)
        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_layers):
            x = F.elu(self.gcs[i](x, edge_index))
            if self.BN: x = self.bns[i](x)
            if self.dropout: x = self.drops[i](x)
        x = pooling_dict[self.pooling](x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)



class GIN_Net(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels,pooling,dropout=0.2):
        super().__init__()

        self.pooling = pooling
        self.dropout = dropout

        self.conv1 = GINConv(
            Sequential(Linear(in_channels, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = pooling_dict[self.pooling](x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x



class GIN_Net1(torch.nn.Module):
    #
    def __init__(self, in_channels, dim, out_channels,pooling,dropout=0.2):
        super().__init__()

        self.pooling = pooling
        self.dropout = dropout

        self.conv1 = GIN(in_channels, dim, num_layers=2, dropout=dropout)
        # self.conv2 = GINConv(dim,dim)

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        # x = self.conv2(x, edge_index)
        x = pooling_dict[self.pooling](x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x

# class Ori_GCN(nn.Module):
#
#     def __init__(self,
#                  in_features,
#                  out_features,
#                  filters=[64, 64, 64],
#                  K=1,
#                  bnorm=False,
#                  n_hidden=0,
#                  dropout=0.2,
#                  adj_sq=False,
#                  scale_identity=False,
#                  reg=False):
#         super(Ori_GCN, self).__init__()
#
#         self.gconv = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
#                                                 out_features=f,
#                                                 K=K,
#                                                 activation=nn.ReLU(inplace=True),
#                                                 bnorm=bnorm,
#                                                 adj_sq=adj_sq,
#                                                 scale_identity=scale_identity) for layer, f in enumerate(filters)]))
#
#         # Fully connected layers
#         fc = []
#         if dropout > 0:
#             fc.append(nn.Dropout(p=dropout))
#         if n_hidden > 0:
#             fc.append(nn.Linear(filters[-1], n_hidden))
#             fc.append(nn.ReLU(inplace=True))
#             if dropout > 0:
#                 fc.append(nn.Dropout(p=dropout))
#             n_last = n_hidden
#         else:
#             n_last = filters[-1]
#         if reg:
#             fc.append(nn.Linear(n_last, 1))
#             fc.append(nn.Sigmoid())
#         else:
#             fc.append(nn.Linear(n_last, out_features))
#
#         self.fc = nn.Sequential(*fc)
#
#     def forward(self, data):
#         gconv_x = self.gconvdata()[0]
#         max_x = torch.max(gconv_x, dim=1)[0].squeeze()  # maxpooling
#         x = self.fc(max_x)
#         return x
#
#     def tiqu_hidden_feature(self, data):
#         gconv_x = self.gconv(data)[0]
#         max_x = torch.max(gconv_x, dim=1)[0].squeeze()
#         x = self.fc(max_x)
#         return gconv_x, max_x, x
#
#     def l2_loss(self):
#         layer = self.layers.children()
#         layer = next(iter(layer))
#         loss = None
#
#         for p in layer.parameters():
#             if loss is None:
#                 loss = p.pow(2).sum()
#             else:
#                 loss += p.pow(2).sum()
#
#         return loss

"code "
class MLP(torch.nn.Module):
    def __init__(self, input, dim, output):
        super().__init__()
        self.fc1 = Linear(input, dim)
        self.fc2 = Linear(dim, dim)
        self.fc3 = Linear(dim, output)

    def forward(self, data):
        x = data.code_x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class MLP1(torch.nn.Module):
    def __init__(self, input, dim, output):
        super().__init__()
        self.fc1 = Linear(input, dim)
        self.fc2 = Linear(dim, output)

    def forward(self, data):
        x = data.code_x
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return F.log_softmax(x, dim=1)

class AutoEncoder(torch.nn.Module):
    def __init__(self,input,dim,output):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, output),
        )

    def forward(self, data):
        x = data.code_x
        encoded = self.encoder(x)
        return F.log_softmax(encoded, dim=-1)


class GCN_Net_1(torch.nn.Module):
    def __init__(self,input_fea,hidden,output_fea):
        super().__init__()
        self.conv1 = GCNConv(input_fea, hidden)
        self.linear = Linear(hidden, output_fea)

    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        # x = F.dropout(x,p=0.1, training=self.training)
        x = self.linear(x).relu()
        return F.log_softmax(x, dim=1)


class GAT_Net_1(torch.nn.Module):
    def __init__(self, in_channels,hidden, out_channels):
        super().__init__()

        self.conv1 = GATConv(in_channels, hidden)
        self.linear = Linear(hidden, out_channels)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x,training=self.training)
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)


# -------------------------------------------------------------

class All_Concat_model_new(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2

    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size, final_size):
        super(All_Concat_model_new, self).__init__()
        self.GCN = GIN_Net2(trans_input_size, trans_hidden, output_size)
        self.MLP = MLP(code_input_size, code_hidden, output_size)
        self.linear = nn.Linear(output_size + output_size, final_size)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, data):
        x, code_x, edge_index, batch = data.x, data.code_x, data.edge_index, data.batch
        code_emb = self.MLP(code_x)
        trans_emb = self.GCN(x, edge_index, batch)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb, trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)

        return F.log_softmax(final_emb, dim=-1)

class All_Concat_model_MLP_GCN(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size,num_layers,pooling,dropout, final_size):
        super(All_Concat_model_MLP_GCN, self).__init__()
        self.GCN = GCN_Net(trans_input_size, trans_hidden, output_size,num_layers,pooling,dropout)
        self.MLP = MLP(code_input_size, code_hidden, output_size)
        self.linear = nn.Linear(output_size + output_size, final_size)

        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, data):
        code_emb = self.MLP(data)
        trans_emb = self.GCN(data)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb, trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)
        final_prob = F.log_softmax(final_emb, dim=-1)
        return code_emb, trans_emb, final_prob

class All_Concat_model_Auto_GCN(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size,num_layers,pooling,dropout, final_size):
        super(All_Concat_model_Auto_GCN, self).__init__()
        self.GCN = GCN_Net(trans_input_size, trans_hidden, output_size,num_layers,pooling,dropout)
        self.Auto = AutoEncoder(code_input_size,code_hidden, output_size)
        self.linear = nn.Linear(output_size + output_size, final_size)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, data):
        code_emb = self.Auto(data)
        trans_emb = self.GCN(data)
        # 这里只是简单的拼接
        concat_emb = torch.cat((code_emb, trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）
        final_emb = self.linear(concat_emb)

        return F.log_softmax(final_emb, dim=-1)


class All_Concat_model_MLP_GCN_attention(nn.Module):
    # trans_input_size, trans_hidden,
    # 交易模块的神经元
    # code_input_size,code_hidden1,code_hidden2,output_size,
    # 代码模块的神经层
    # output_size 是共同的输出
    # final_size 最终层 类别数 final_size  2
    def __init__(self, trans_input_size, trans_hidden, code_input_size, code_hidden, output_size,num_layers,pooling,dropout, final_size):
        super(All_Concat_model_MLP_GCN_attention, self).__init__()
        self.GCN = GCN_Net(trans_input_size, trans_hidden, output_size,num_layers,pooling,dropout)
        self.MLP = MLP(code_input_size, code_hidden, output_size)
        self.attn = nn.Linear(output_size * 2, output_size)
        self.linear_a = nn.Linear(output_size, final_size)
        self.linear = nn.Linear(output_size*2, final_size)


        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, data):
        code_emb = self.MLP(data)
        trans_emb = self.GCN(data)
        # 这里只是简单的拼接 # todo attention 拼接
        concat_emb = torch.cat((code_emb, trans_emb), 1)  # 按维数0拼接（竖着拼） #按维数1拼接（横着拼）

        final_emb = self.linear(concat_emb)

        return F.log_softmax(final_emb, dim=-1)




