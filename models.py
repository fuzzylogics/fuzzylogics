from re import I
import numpy as np
from numpy.core.fromnumeric import compress
from six import b
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid, ELU, Softmax, BatchNorm1d, Identity
from torch_geometric.nn import MessagePassing, knn_graph

from opt_einsum import contract




class CNN_GGA_0(nn.Module):
    def __init__(self):
        super(CNN_GGA_0, self).__init__()
        self.rho_type = "GGA"
        self.conv1 = nn.Conv3d(4, 8, 4) # 9x9x9 -> 6x6x6
        self.fc1 = nn.Linear(216, 108)
        self.fc2 = nn.Linear(108, 50)
        self.fc3 = nn.Linear(50, 25)
        self.fc4 = nn.Linear(25, 1)

    def forward(self, x):
        # x shape: 4 x 9 x 9 x 9
        # for GGA-like NN, use electron density and its gradients

        x = F.max_pool3d(F.elu(self.conv1(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CNN_GGA_1(nn.Module):
    def __init__(self, bias=True):
        super(CNN_GGA_1, self).__init__()
        self.rho_type = "GGA"
        self.conv1 = nn.Conv3d(4, 8, 4, bias=bias) # 4@9x9x9 -> 8@6x6x6, 4x4x4 kernel
        self.conv2 = nn.Conv3d(8, 16, 3, bias=bias) # 8@6x6x6 -> 16@4x4x4, 3x3x3 kernel
        self.fc1 = nn.Linear(128, 64, bias=bias)
        self.fc2 = nn.Linear(64, 32, bias=bias)
        self.fc3 = nn.Linear(32, 16, bias=bias)
        self.fc4 = nn.Linear(16, 1, bias=bias)

    def forward(self, x):
        # x shape: 4 x 9 x 9 x 9
        # for GGA-like NN, use electron density and its gradients

        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.max_pool3d(x, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CNN_GGA_2(nn.Module):
    def __init__(self, bias=True, act=None):
        super().__init__()
        #<name>abc: block_layer:a, block:b, layer_in_block:c

        self.conv111 = nn.Conv3d(4, 8, 3, bias=bias)
        self.conv112 = nn.Conv3d(8, 16, 3, bias=bias)
        self.conv113 = nn.Conv3d(16, 32, 3, bias=bias)

        self.conv121 = nn.Conv3d(4, 8, 3, dilation=2, bias=bias)
        self.conv122 = nn.Conv3d(8, 16, 3, bias=bias)

        self.conv131 = nn.Conv3d(4, 8, 3, bias=bias)
        self.conv132 = nn.Conv3d(8, 16, 3, dilation=2, bias=bias)

        self.conv211 = nn.Conv3d(64, 32, 1, bias=bias)
        self.conv212 = nn.Conv3d(32, 16, 1, bias=bias)

        self.conv221 = nn.Conv3d(64, 32, 2, bias=bias)
        self.conv222 = nn.Conv3d(32, 16, 2, bias=bias)

        self.conv231 = nn.Conv3d(64, 32, 3, bias=bias)

        self.fc241 = nn.Linear((32+16+16)*27, 128, bias=bias)
        
        self.fc311 = nn.Linear(16*27+16*1+32*1+128, 128, bias=bias)
        self.fc312 = nn.Linear(128, 64, bias=bias)
        self.fc313 = nn.Linear(64, 32, bias=bias)
        self.fc314 = nn.Linear(32, 16, bias=bias)
        self.fc315 = nn.Linear(16, 1, bias=bias)

        self.act = F.elu if act==None else act

    def forward(self, x0):
        x111 = self.conv111(x0)
        x112 = self.conv112(x111)
        x113 = self.conv113(x112)
        x121 = self.conv121(x0)
        x122 = self.conv122(x121)
        x131 = self.conv131(x0)
        x132 = self.conv132(x131)
        x1 = torch.cat((x113, x122, x132), dim=1)
        x1_lin = x1.view((-1, self.n(x1)))
        x211 = self.conv211(x1)
        x212 = self.conv212(x211)
        x21_lin = x212.view((-1, self.n(x212)))
        x221 = self.conv221(x1)
        x222 = self.conv222(x221)
        x22_lin = x222.view((-1, self.n(x222)))
        x231 = self.conv231(x1)
        x23_lin = x231.view((-1, self.n(x231)))
        x241 = self.act(self.fc241(x1_lin))
        x2 = torch.cat((x21_lin, x22_lin, x23_lin, x241), dim=1)
        x311 = self.act(self.fc311(x2))
        x312 = self.act(self.fc312(x311))
        x313 = self.act(self.fc313(x312))
        x314 = self.act(self.fc314(x313))
        x315 = self.fc315(x314)
        y = x315
        return y
 
    def n(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CNN_GGA_3(nn.Module):
    def __init__(self, icpt_channels=128, compress_channels=16, bias=True, act=None):
        super().__init__()
        self.conv_block = conv_block_CNN_GGA_3(out_channels=icpt_channels, bias=bias)
        self.icpt_block1 = icpt_block_CNN_GGA_3(icpt_channels, icpt_channels, bias=bias)
        self.icpt_block2 = icpt_block_CNN_GGA_3(icpt_channels, icpt_channels, bias=bias)
        self.icpt_block3 = icpt_block_CNN_GGA_3(icpt_channels, icpt_channels, bias=bias)
        self.compress_block = nn.Conv3d(icpt_channels, compress_channels, kernel_size=1, bias=bias)
        self.fc_block = fc_block_CNN_GGA_3(compress_channels*5*5*5, bias=bias, act=act)
    def forward(self, x):
        x = self.conv_block(x)
        # print('con', x.shape)
        x = self.icpt_block1(x)
        # print('i1', x.shape)
        x = self.icpt_block2(x)
        # print('i2', x.shape)
        x = self.icpt_block3(x)
        # print('i3', x.shape)
        x = self.compress_block(x)
        # print('cp', x.shape)
        x = x.view(-1, self.n(x))
        x = self.fc_block(x)
        # print('out', x.shape)
        return x
    def n(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class conv_block_CNN_GGA_3(nn.Module):
    def __init__(self, out_channels=256, bias=True):
        super().__init__()
        self.conv1 = nn.Conv3d(4, 32, 3, bias=bias)
        self.conv2 = nn.Conv3d(32, 32, 1, bias=bias)
        self.conv3 = nn.Conv3d(32, out_channels, 3, bias=bias)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class icpt_block_CNN_GGA_3(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, bias=True):
        super().__init__()
        self.conv1_1 = nn.Conv3d(in_channels, out_channels//2, kernel_size=1, bias=bias)
        self.conv1_2 = nn.Conv3d(in_channels, out_channels//4, kernel_size=3, padding=1, bias=bias)
        self.conv1_3 = nn.Conv3d(in_channels, out_channels//4, kernel_size=5, padding=2, bias=bias)
        self.pool1_4 = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x_conv1_1 = self.conv1_1(x)
        x_conv1_2 = self.conv1_2(x)
        x_conv1_3 = self.conv1_3(x)
        x_pool1_4 = self.pool1_4(x)
        # print( x_conv1_1.shape)
        # print( x_conv1_2.shape)
        # print( x_conv1_3.shape)
        # print( x_pool1_4.shape)
        return torch.cat([x_conv1_1, x_conv1_2, x_conv1_3], 1) + x_pool1_4

class fc_block_CNN_GGA_3(nn.Module):
    def __init__(self, in_channels, bias=True, act=None):
        super().__init__()
        # self.fc1 = nn.Linear(in_channels, 512, bias=bias)
        self.fc1 = nn.Linear(in_channels, 128, bias=bias)
        self.fc2 = nn.Linear(128, 32, bias=bias)
        self.fc3 = nn.Linear(32, 4, bias=bias)
        self.fc4 = nn.Linear(4, 1, bias=bias)
        self.act = F.elu if act==None else act
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.fc4(x)
        return x


class CNN_GGA_4(nn.Module):
    def __init__(self, bias=True, activation='sig'):
        super().__init__()
        self.conv1 = nn.Conv3d(4, 8, 4, bias=bias)
        self.conv2 = nn.Conv3d(8, 16, 3, bias=bias)
        self.pool1 = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(128, 64, bias=bias)
        self.fc2 = nn.Linear(64, 32, bias=bias)
        self.fc3 = nn.Linear(32, 16, bias=bias)
        self.fc4 = nn.Linear(16, 1, bias=bias)
        self.activation_shift = 0.0
        if activation == 'sig': 
            self.activation = Sigmoid()
            self.activation_shift = -0.5
        elif activation == 'elu':
            self.activation = ELU()
        else: raise NotImplementedError

    def forward(self, x):
        print('0', x.min().item(), x.max().item())
        x = self.activation(self.conv1(x)) + self.activation_shift
        print('1', x.min().item(), x.max().item())
        x = self.activation(self.conv2(x)) + self.activation_shift
        print('2', x.min().item(), x.max().item())
        x = self.pool1(x)
        x = x.view(-1, self.n(x))
        x = self.activation(self.fc1(x)) + self.activation_shift
        print('3', x.min().item(), x.max().item())
        x = self.activation(self.fc2(x)) + self.activation_shift
        print('4', x.min().item(), x.max().item())
        x = self.activation(self.fc3(x)) + self.activation_shift
        print('5', x.min().item(), x.max().item())
        x = self.fc4(x)
        return x

    def n(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CNN_GGA_5(nn.Module):
    def __init__(self, bias=True, activation='sig'):
        super().__init__()
        self.icpt_block1 = self.icpt_block_CNN_GGA_5(4, icpt_channels=4, bias=bias) # out_channels=4*6+4=28
        # self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=0)
        # self.compress_block1_1 = nn.Conv3d(self.icpt_block1.out_channels, 
        #                                    out_channels = self.icpt_block1.out_channels * 16 // 7,
        #                                    kernel_size=3, stride=2, padding=0,
        #                                    groups = self.icpt_block1.out_channels // 7,
        #                                    bias=bias)
        self.compress_block1_1 = nn.Conv3d(self.icpt_block1.out_channels, 
                                           out_channels = self.icpt_block1.out_channels * 16 // 7,
                                           kernel_size=3, stride=2, padding=0,
                                           groups = 1,
                                           bias=bias)
        self.compress_block1_2 = nn.Conv3d(64, out_channels=16, 
                                           kernel_size=1, stride=1, padding=0, bias=bias)
        self.icpt_block2 = self.icpt_block_CNN_GGA_5(16, icpt_channels=16, bias=bias)
        # self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.compress_block2_1 = nn.Conv3d(self.icpt_block2.out_channels, 
                                           out_channels = self.icpt_block2.out_channels * 16 // 7,
                                           kernel_size=2, stride=2, padding=0,
                                           groups = 1,
                                           bias=bias)
        self.compress_block2_2 = nn.Conv3d(256, out_channels=64, 
                                           kernel_size=1, bias=bias)
        self.fc_block = self.fc_block_CNN_GGA_5(64*2**3, bias=bias, activation=activation)
        self.determine_activation(activation)
    def forward(self, x):
        x = self.activation(self.icpt_block1(x))
        # x = self.pool1(x)
        x = self.activation(self.compress_block1_1(x))
        x = self.activation(self.compress_block1_2(x))
        x = self.activation(self.icpt_block2(x))
        # x = self.pool2(x)
        x = self.activation(self.compress_block2_1(x))
        x = self.activation(self.compress_block2_2(x))      
        x = x.view(-1, self.n(x))
        x = self.fc_block(x)
        return x
    def n(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    def determine_activation(self, activation):
        if activation == 'sig': self.activation = ShiftedSigmoid()
        elif activation == 'elu': self.activation = ELU()
        else: raise NotImplementedError

    class icpt_block_CNN_GGA_5(nn.Module):
        def __init__(self, in_channels=4, icpt_channels=4, bias=True):
            super().__init__()
            self.conv_1 = nn.Conv3d(in_channels, icpt_channels, kernel_size=1, bias=bias)
            self.conv_3 = nn.Conv3d(in_channels, icpt_channels, kernel_size=3, padding=1, bias=bias)
            self.conv_5 = nn.Conv3d(in_channels, icpt_channels, kernel_size=5, padding=2, bias=bias)
            self.conv_x = nn.Conv3d(in_channels, icpt_channels, kernel_size=(7,1,1), padding=(3,0,0), bias=bias)
            self.conv_y = nn.Conv3d(in_channels, icpt_channels, kernel_size=(1,7,1), padding=(0,3,0), bias=bias)
            self.conv_z = nn.Conv3d(in_channels, icpt_channels, kernel_size=(1,1,7), padding=(0,0,3), bias=bias)
            self.pool = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)
            self.out_channels = icpt_channels*6 + in_channels
        def forward(self, x):
            y = []
            y += [self.conv_1(x)]
            y += [self.conv_3(x)]
            y += [self.conv_5(x)]
            y += [self.conv_x(x)]
            y += [self.conv_y(x)]
            y += [self.conv_z(x)]
            y += [self.pool(x)]
            return torch.cat(y, 1)

    class fc_block_CNN_GGA_5(nn.Module):
        def __init__(self, in_channels, bias=True, activation='sig'):
            super().__init__()
            # self.fc1 = nn.Linear(in_channels, 512, bias=bias)
            self.fc1 = nn.Linear(in_channels, 128, bias=bias)
            self.fc2 = nn.Linear(128, 32, bias=bias)
            self.fc3 = nn.Linear(32, 4, bias=bias)
            self.fc4 = nn.Linear(4, 1, bias=bias)
            self.activation_shift = 0.0
            self.determine_activation(activation)
        def forward(self, x):
            x = self.activation(self.fc1(x)) + self.activation_shift
            x = self.activation(self.fc2(x)) + self.activation_shift
            x = self.activation(self.fc3(x)) + self.activation_shift
            x = self.fc4(x)
            return x
        def determine_activation(self, activation):
            if activation == 'sig': self.activation = ShiftedSigmoid()
            elif activation == 'elu': self.activation = ELU()
            else: raise NotImplementedError

# class cmps_block_CNN_GGA_5(nn.Module):
#     def __init__(self, in_channels, compress_channels, pool_size=2, bias=True, act=None):
#         if pool_size > 1: self.pool = nn.AvgPool3d(kernel_size=pool_size, stride=pool_size)
#         self.conv = nn.Conv3d(in_channels=in_channels, out_channels=compress_channels, kernel_size=1)
#     def forward(self, x):
#         if getattr(self, 'pool', None) is not None: x = self.pool(x)
#         x = self.conv(x)
#         return x


class ShiftedSigmoid(nn.Module):
    def __init__(self, shift=-0.5):
        super().__init__()
        self.shift = shift
        self.sigmoid = Sigmoid()
    def forward(self, x):
        return self.sigmoid(x) + self.shift


class CNN_GGA_all_sym_2(nn.Module):
    def __init__(self):
        super(CNN_GGA_all_sym_2, self).__init__()
        self.rho_type = "GGA"
        self.conv1 = nn.Conv3d(4, 8, 4) # 4@9x9x9 -> 8@6x6x6, 4x4x4 kernel
        self.conv2 = nn.Conv3d(8, 16, 3) # 8@6x6x6 -> 16@4x4x4, 3x3x3 kernel
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        # x shape: ni x nr x 4 x 9 x 9 x 9
        ni = x.size()[1]
        for i in range(ni):
            xi = x[:, i]
            xi = F.elu(self.conv1(xi))
            xi = F.elu(self.conv2(xi))
            xi = F.max_pool3d(xi, 2)
            xi = xi.view(-1, self.num_flat_features(xi))      
            xi = F.elu(self.fc1(xi))
            if i == 0:
                xm = xi.unsqueeze(1).repeat(1,ni,1)
            if i == 1:
                err = torch.sum((xi - xm[:,0,:])**2, 1, keepdim=True)
            if i > 1:
                for j in range(i):
                    err += torch.sum((xi - xm[:,j,:])**2, 1, keepdim=True)    
            xm[:,i,:] = xi
            xi = F.elu(self.fc2(xi))
            xi = F.elu(self.fc3(xi))
            xi = self.fc4(xi)
            if i == 0:
                y = xi
            else:
                y += xi         
        
        #y = torch.unsqueeze(y, 0)
        #print(y.shape, err.shape)
        return y/ni, 2*err/ni/(ni-1)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class GAT_dm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn0 = BatchNorm1d(1)
        self.conv0 = MultiHeadAttentionGraphLayer(1, 6, 2, 6, n_heads=3)
        self.bn1 = BatchNorm1d(6*3)
        self.conv1 = MultiHeadAttentionGraphLayer(1+6*3, 6, 0, 0, n_heads=3)
        self.bn2 = BatchNorm1d(6*3)
        self.conv2 = MultiHeadAttentionGraphLayer(1+6*3+6*3, 6, 0, 0, n_heads=3)
        # self.conv4 = MultiHeadAttentionGraphLayer(64*2, 32, 0, 0, n_heads=2)
        # self.conv5 = MultiHeadAttentionGraphLayer(32*2, 16, 0, 0, n_heads=2)
        self.bn3 = BatchNorm1d(6*3)
        self.fcc = Seq(Linear(6*3, 1),
                       )

    def forward(self, x, edge_index, e):
        print(x.shape, edge_index.shape, e.shape)
        x0 = self.bn0(x)
        x1 = self.conv0(x0, edge_index, e)
        x1 = self.bn1(x1)
        x1 = torch.cat((x0, x1), 1)
        x2 = self.conv1(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = torch.cat((x1, x2), 1)
        y = self.conv2(x2, edge_index)
        # x = self.conv4(x, edge_index)
        # x = self.conv5(x, edge_index)
        y = self.bn3(y)
        y = self.fcc(y)
        return y


class MP_layer_dm(torch.nn.Module):
    def __init__(self, n_node_feats, n_edge_feats, n_message_feats, n_out_feats):
        super().__init__()
        self.k = None
        self.node_encoding = Identity()
        self.psi = Seq(Linear(2*n_node_feats+n_edge_feats, n_message_feats), ELU())
        self.gamma = Seq(Linear(n_node_feats + n_message_feats, n_out_feats), )
    
    def forward(self, x, edge_index, e, agg='mean', k=None):
        if k is not None: assert edge_index.shape[1] // x.shape[0] == k
        self.k = edge_index.shape[1] // x.shape[0]
        x = self.node_encoding(x)
        # print(x.shape, edge_index.shape)
        source, drain = x[edge_index[0]], x[edge_index[1]]
        mess = self.message(source, drain, e)
        x = self.aggregate(x, mess, agg)
        return x
        
    def message(self, source, drain, e):
        return self.psi(torch.cat((source, drain, e), 1))

    def aggregate(self, x, mess, agg):        
        if agg == 'mean': agg_func = torch.mean
        all_mess = agg_func(mess.reshape((-1,self.k)+mess.shape[1:]), 1)
        # print('mess',mess.shape,'all_mess',all_mess.shape, 'x', x.shape)
        return self.gamma(torch.cat((x, all_mess), 1))
        

class MP_2l_dm(torch.nn.Module):
    def __init__(self, n_node_feats, n_edge_feats, n_message1_feats, n_out1_feats, n_message2_feats):
        super().__init__()
        self.mp1 = MP_layer_dm(n_node_feats, n_edge_feats, n_message1_feats, n_out1_feats)
        self.mp2 = MP_layer_dm(n_out1_feats, n_edge_feats, n_message2_feats, 1)

    def forward(self, x, edge_index, e):
        x1 = self.mp1(x, edge_index, e)
        x2 = self.mp2(x1, edge_index, e)
        return x2.squeeze(1)


# test simplist MLP
class MLP(nn.Module):
    def __init__(self, nh):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(9*9*9*4, nh),
            nn.ReLU(),
            nn.Linear(nh, 1)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x



# graph models


# class MultiHeadAttentionGraphLayer(torch.nn.Module):
#     def __init__(self, x_channels, x_emb_channels, e_channels, e_emb_channels, n_heads, bias=True):
#         super(MultiHeadAttentionGraphLayer, self).__init__()
#         self.n_heads = n_heads
#         self.node_embedding = Linear(x_channels, x_emb_channels*n_heads, bias=bias)
#         # self.node_embedding = Seq(Linear(x_channels, x_emb_channels),
#         #                     ELU(),
#         #                     Linear(x_emb_channels, x_emb_channels))
#         # self.source_embedding = Seq(Linear(x_channels, x_emb_channels),
#         #                     ELU(),
#         #                     Linear(x_emb_channels, x_emb_channels))
#         # self.drain_embedding = Seq(Linear(x_channels, x_emb_channels),
#         #                     ELU(),
#         #                     Linear(x_emb_channels, x_emb_channels))
#         if e_channels > 0:
#             self.edge_embedding = Linear(e_channels, e_emb_channels*n_heads, bias=bias)
#         # self.edge_embedding = Seq(Linear(e_channels, e_emb_channels),
#         #                     ELU(),
#         #                     Linear(e_emb_channels, e_emb_channels))
#         self.attention = Seq(Linear(2*x_emb_channels+e_emb_channels, 1, bias=bias),
#                             ELU(),
#                             Softmax(dim=1))
#         self.aggregation_function = ELU()

#     def forward(self, x, edge_index, e=None, k=None, head_agg='cat'):
#         if k is None: k = edge_index.shape[1] // x.shape[0]
#         source_index, drain_index = edge_index[0], edge_index[1]
#         x = self.node_embedding(x)
#         source_features = x[source_index].view(-1, k, self.n_heads, x.shape[1]//self.n_heads)  # shape: (N_nodes, k, n_heads, x_emb_channels)
#         drain_features = x[drain_index].view(-1, k, self.n_heads, x.shape[1]//self.n_heads)  # shape: (N_nodes, k, n_heads, x_emb_channels)
#         if e is None:
#             edge_features = torch.tensor([], dtype=x.dtype, device=x.device)
#         else:
#             e = self.edge_embedding(e)
#             edge_features = e.view(-1, k, self.n_heads, e.shape[1]//self.n_heads)  # shape: (N_nodes, k, n_heads, e_emb_channels)
#         # edge_features = e.view(-1, k, self.n(e)).unsqueeze(2).repeat(1,1,self.n_heads,1)  # edge w/o embedding, shape: (N_nodes, k, n_heads, e_channels)
#         att = self.attention(torch.cat((source_features, drain_features, edge_features), 3)).squeeze(3)
#         if head_agg == 'avg':
#             y = contract('nkh, nkhf -> nf', att, source_features) / k / self.n_heads
#         elif head_agg == 'cat':
#             y = contract('nkh, nkhf -> nhf', att, source_features) / k
#             y = y.view(-1, self.n_heads*y.shape[2])
#         y = self.aggregation_function(y)
#         return y

# updated version Jul20,2021
class MultiHeadAttentionGraphLayer(torch.nn.Module):
    def __init__(self, 
                 x_channels, x_emb_channels, 
                 e_channels=0, e_emb_channels=3, e_emb=None,
                 n_heads=3, 
                 attention='lin',
                 bias=True, activation=None):
        super(MultiHeadAttentionGraphLayer, self).__init__()
        
        self.n_heads = n_heads
        self.x_emb_channels = x_emb_channels
        self.node_embedding = Linear(x_channels, x_emb_channels*n_heads, bias=bias)
        # activation
        if activation is None: self.activation = Identity()
        elif activation == 'elu': self.activation = ELU()
        elif activation == 'sig': self.activation = Sigmoid()
        # edge embedding
        if e_channels > 0:
            if e_emb is None: 
                self.e_emb_channels = e_channels
                print('e_emb_channels not in use, being set to e_channels %d.'%(e_channels))
                self.edge_embedding = Identity()
            elif e_emb == 'lin' and e_emb_channels > 0:
                self.e_emb_channels = e_emb_channels
                print("e_emb_channels = %d"%(e_emb_channels))
                self.edge_embedding = Linear(e_channels, e_emb_channels*n_heads, bias=bias)                
        else: 
            self.e_emb_channels = 0
            print('e_emb_channels not in use, being set to 0.')
        # attention
        if attention == 'lin':                
            self.attention = Seq(Linear(2*x_emb_channels+e_emb_channels, 1, bias=bias),
                                 self.activation,
                                 Softmax(dim=1))
        elif attention == 'inp':
            self.attention = self.inner_product_attention
        self.aggregation_function = self.activation
    
    def inner_product_attention(self, data_s_d_e, edge4weights=False):
        source_features = data_s_d_e[:, :, :, :self.x_emb_channels]
        drain_features =  data_s_d_e[:, :, :, self.x_emb_channels:2*self.x_emb_channels] 
        edge_features = data_s_d_e[:, :, :, 2*self.x_emb_channels:]
        if edge4weights:
            assert len(edge_features) > 0
            eps = 1.0e-8
            w = 1.0 / ( (edge_features**2).sum(-1) + eps )
        else: w = 1.0
        att = w * contract('nkhf, nkhf -> nkh', source_features, drain_features)
        return att

    def forward(self, x, edge_index, e=None, k=None, head_agg='cat'):
        if k is None: k = edge_index.shape[1] // x.shape[0]
        source_index, drain_index = edge_index[0], edge_index[1]
        x = self.node_embedding(x)
        source_features = x[source_index].view(-1, k, self.n_heads, x.shape[1]//self.n_heads)  # shape: (N_nodes, k, n_heads, x_emb_channels)
        drain_features = x[drain_index].view(-1, k, self.n_heads, x.shape[1]//self.n_heads)  # shape: (N_nodes, k, n_heads, x_emb_channels)
        if e is None or self.e_emb_channels == 0:
            edge_features = torch.tensor([], dtype=x.dtype, device=x.device)
        else:
            e = self.edge_embedding(e)
            if e.shape[-1] == self.n_heads * self.e_emb_channels:
                edge_features = e.view(-1, k, self.n_heads, self.e_emb_channels)  # shape: (N_nodes, k, n_heads, e_emb_channels)
            elif e.shape[-1] == self.e_emb_channels:
                edge_features = e.view(-1, k, 1, self.e_emb_channels).expand(-1, -1, self.n_heads, -1)
        # edge_features = e.view(-1, k, self.n(e)).unsqueeze(2).repeat(1,1,self.n_heads,1)  # edge w/o embedding, shape: (N_nodes, k, n_heads, e_channels)
        att = self.attention(torch.cat((source_features, drain_features, edge_features), 3)).squeeze(3)
        
        if head_agg == 'avg':
            y = contract('nkh, nkhf -> nf', att, source_features) / k / self.n_heads
        elif head_agg == 'cat':
            y = contract('nkh, nkhf -> nhf', att, source_features) / k
            y = y.view(-1, self.n_heads*y.shape[2])
        y = self.aggregation_function(y)
        return y


class GAT_GGA(torch.nn.Module):
    def __init__(self, use_bn=False, bias=True):
        super().__init__()
        self.use_bn = use_bn
        if use_bn: self.bn0 = BatchNorm1d(4)
        self.conv0 = MultiHeadAttentionGraphLayer(4, 6, 3, 3, n_heads=6, bias=bias)
        if use_bn: self.bn1 = BatchNorm1d(6*6)
        self.conv1 = MultiHeadAttentionGraphLayer(4+6*6, 6, 0, 0, n_heads=3, bias=bias)
        if use_bn: self.bn2 = BatchNorm1d(6*3)
        self.conv2 = MultiHeadAttentionGraphLayer(4+6*6+6*3, 6, 0, 0, n_heads=3, bias=bias)
        if use_bn: self.bn3 = BatchNorm1d(6*3)
        # self.conv4 = MultiHeadAttentionGraphLayer(64*2, 32, 0, 0, n_heads=2)
        # self.conv5 = MultiHeadAttentionGraphLayer(32*2, 16, 0, 0, n_heads=2)
        self.fcc = Seq(Linear(6*3, 1, bias=bias),
                       ELU())

    def forward(self, x0, edge_index, e):
        if self.use_bn: x0 = self.bn0(x0)
        x1 = self.conv0(x0, edge_index, e)
        if self.use_bn: x1 = self.bn1(x1)
        x1 = torch.cat((x0, x1), 1)
        x2 = self.conv1(x1, edge_index)
        if self.use_bn: x2 = self.bn2(x2)
        x2 = torch.cat((x1, x2), 1)
        y = self.conv2(x2, edge_index)
        # x = self.conv4(x, edge_index)
        # x = self.conv5(x, edge_index)
        if self.use_bn: y = self.bn3(y)
        y = self.fcc(y)
        return y


class GeneralGraphConv(MessagePassing):
    def __init__(self, in_channels, mid_channels, out_channels,
                 edge_in_channels, edge_mid_channels, edge_out_channels,
                 bias=True):
        super(GeneralGraphConv, self).__init__(aggr='add')
        self.mlp_node = Seq(Linear(2*in_channels+edge_in_channels, mid_channels, bias=bias),
                       ELU(),
                       Linear(mid_channels, out_channels, bias=bias))
        # self.node2edge = Seq(Linear(out_channels, out_channels, bias=bias),
        #                ELU(),
        #                Linear(out_channels, out_channels, bias=bias))
        # self.mlp_edge = Seq(Linear(out_channels+edge_in_channels, edge_mid_channels, bias=bias),
        #                ELU(),
        #                Linear(edge_mid_channels, edge_out_channels, bias=bias))

    def forward(self, edge_index, x, e):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # print(x.shape)
        # print(x)
        x = self.propagate(edge_index, x=x, e=e)
        # print(x[edge_index[0],:].shape)

        # e = torch.cat((torch.abs(self.node2edge(x[edge_index[0],:]) - self.node2edge(x[edge_index[1],:])), e), dim=1)
        # e = self.mlp_edge(e)
        return x, e

    def message(self, x_i, x_j, e):
        # x_i and x_j has shape [E, in_channels]
        # print(x_i.shape, x_j.shape, e.shape)
        tmp = torch.cat((x_i, x_j, e), dim=1)
        # print("tmp before: ", tmp)
        tmp = self.mlp_node(tmp)
        # print("tmp after", tmp)
        return tmp

class StaticGeneralGraphConv_GGA(torch.nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.conv1 = GeneralGraphConv(4 ,4 ,8 ,3 ,3, 3, bias=bias)
        self.conv2 = GeneralGraphConv(8 ,8 ,8 ,3, 3, 3, bias=bias)
        # self.conv3 = GeneralGraphConv(8 ,8 ,8 ,10, 10, 10)
        # self.conv4 = GeneralGraphConv(8 ,8 ,8 ,10, 10, 10)
        self.conv5 = GeneralGraphConv(8 ,4, 1 ,3 ,3, 1, bias=bias)

    def forward(self, x, edge_index=None, e=None, coords=None, k=9):
        if edge_index is None:
            print("Generating graph again ...")
            edge_index = knn_graph(coords, k, loop=False, flow=self.conv1.flow)
            e = torch.abs(coords[edge_index[1],:] - coords[edge_index[0],:])
        eps = 1.0e-8
        e = e / ((e**2).sum(-1, keepdim=True) + eps)
        x, e = self.conv1(edge_index, x, e)
        # print('1, x', x.shape, 'e', e.shape)
        x, e = self.conv2(edge_index, x, e)
        # print('2, x', x.shape, 'e', e.shape)
        # x, e = self.conv3(edge_index, x, e)
        # x, e = self.conv4(edge_index, x, e)
        x, e = self.conv5(edge_index, x, e)
        # print('3, x', x.shape, 'e', e.shape)
        #print(x.shape)
        return x, e


class FC_GGA(torch.nn.Module):
    def __init__(self, n_feats=4*9*9*9, n_encoded=None, bias=True, activation='sig'):
        super().__init__()
        self.n_encoded = n_encoded
        if n_encoded is not None:
            self.trans = nn.Conv3d(4, n_encoded, 1)
            self.lin1 = Linear(n_encoded*9*9*9, 128, bias=bias)
        else: self.lin1 = Linear(4*9*9*9, 128, bias=bias)
        self.lin2 = Linear(128, 32, bias=bias)
        self.lin3 = Linear(32, 8, bias=bias)
        self.lin4 = Linear(8, 1, bias=bias)
        self.activation_shift = 0.0
        if activation == 'sig': 
            self.activation = Sigmoid()
            self.activation_shift = -0.5
        elif activation == 'elu': self.activation = ELU()
        else: raise NotImplementedError

    def forward(self, x):
        if self.n_encoded is not None:
            x = self.trans(x)
        x = x.view(-1, self.n(x))
        # print('1', x.min().item(), x.max().item())
        x = self.activation(self.lin1(x)) + self.activation_shift
        # print('2', x.min().item(), x.max().item())
        x = self.activation(self.lin2(x)) + self.activation_shift
        # print('3', x.min().item(), x.max().item())
        x = self.activation(self.lin3(x)) + self.activation_shift
        # print('4', x.min().item(), x.max().item())
        x = self.lin4(x)
        # print('5', x.min().item(), x.max().item())
        return x
    
    def n(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class GaussFiltersGGA(torch.nn.Module):
    def __init__(self, n_gauss=1000, cube_len=1.0, n_points=9, bias=True, activation='sig'):
        super().__init__()
        self.n_gauss = n_gauss
        self.cube_coords = self.cube(cube_len, n_points)
        self.register_buffer('cube_coords_consts', self.cube_coords)
        # self.gauss_params = Linear(4*n_points**3, n_gauss*3*2)
        self.mu = nn.Parameter(torch.zeros((n_gauss,3), requires_grad=True, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.ones((n_gauss,3), requires_grad=True, dtype=torch.float32))
        self.fc = FC_GGA(n_feats=4*n_gauss, bias=bias, activation=activation)

    def cube(self, l, n):
        x = np.linspace(-l/2, l/2, n) 
        x, y, z = np.meshgrid(x, x, x, indexing='ij')
        coords = np.stack((x, y, z), -1)
        return torch.from_numpy(coords).float()
    # def cube(self, cube_len, n_points):
    #     one_edge = np.linspace(-cube_len/2, cube_len/2, n_points, endpoint=True)
    #     one_edge = torch.tensor(one_edge, dtype=float)
    #     dists = torch.cat((one_edge.view(n_points, 1, 1, 1).expand(n_points, n_points, n_points, -1),
    #                        one_edge.view(1, n_points, 1, 1).expand(n_points, n_points, n_points, -1),
    #                        one_edge.view(1, 1, n_points, 1).expand(n_points, n_points, n_points, -1))
    #                       -1)
    #     return dists

    # def gauss_filter(self, x, mu, sigma):
    #     device = x.device
    #     d = self.cube_coords.to(device)
    #     d_shape = d.shape[:-1]
    #     assert d.shape[-1] == 3
    #     g_shape = mu.shape[:-1]
    #     assert g_shape == sigma.shape[:-1]
    #     assert mu.shape[-1] == 3 and sigma.shape[-1] == 3
    #     d = d.view((1,)*len(g_shape)+d_shape+(3,))
    #     print('d', d.shape)
    #     mu = mu.view(g_shape+(1,)*len(d_shape)+(3,))
    #     print('mu', mu.shape)
    #     sigma = sigma.view(g_shape+(1,)*len(d_shape)+(3,))
    #     print('sigma', sigma.shape)
    #     g = torch.empty(g_shape, dtype=float, device=device)
    #     for i in range(0, len(x), 100):
    #         print(i)
    #         bg, ed = i, max(i+100, len(x))
    #         filter = ((d - mu[bg:ed])**2 / sigma[bg:ed]).sum(-1).exp()
    #         filter = filter / filter.sum((-1,-2,-3), keepdim=True)
    #         print('filter', filter.shape)
    #         # g[bg:ed] = (x[bg:ed] * filter).sum(-1,-2,-3)
    #         g[bg:ed] = contract('idxyz,ifxyz->ifd', x[bg:ed], filter)
    #     print('g', g.shape)
    #     return g

    def gauss_filter(self, x, mu, sigma):
        d = self.cube_coords_consts
        d_shape = d.shape[:-1]
        assert d.shape[-1] == 3
        g_shape = mu.shape[:-1]
        assert g_shape == sigma.shape[:-1]
        assert mu.shape[-1] == 3 and sigma.shape[-1] == 3
        d = d.view((1,)*len(g_shape)+d_shape+(3,))
        # print('d', d.shape)
        mu = mu.view(g_shape+(1,)*len(d_shape)+(3,))
        # print('mu', mu.shape)
        sigma = sigma.view(g_shape+(1,)*len(d_shape)+(3,))
        # print('sigma', sigma.shape)
        f = ((d - mu)**2 / sigma).sum(-1).exp()
        g = contract('idxyz, fxyz -> idf', x, f)
        # print('g', g.shape)
        return g.reshape(-1,self.n(g))

    def forward(self, x):
        # # print('__', self.gauss_params(x.view(-1,self.n(x))).shape)
        # mu, sigma = self.gauss_params(x.view(-1,self.n(x)))\
        #             .view((-1,self.n_gauss,3,2)).split(1, dim=-1)
        # mu, sigma = mu.squeeze(-1), sigma.squeeze(-1)
        # # print('mu', mu.shape, 'sigma', sigma.shape)

        # added dims: n_samples at the beginning and n_gauss at the end, respectively
        # x = x.unsqueeze(-1) * self.gauss_filter(mu, sigma).unsuqeeze(0)
        print('mu', self.mu.mean().item(), self.mu.min().item(), self.mu.max().item())
        print('sigma', self.sigma.mean().item(), self.sigma.min().item(), self.sigma.max().item())
        x = self.gauss_filter(x, self.mu, self.sigma)
        # print('x', x.shape)
        # x = x.sum((2, 3, 4)) # sum over box dims
        x = x.view(-1, self.n(x))
        x = self.fc(x)
        return x
        
    def n(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class SimpleInception(torch.nn.Module):
    def __init__(self, kernel_len=9, n_encoded=8, bias=True, activation='sig'):
        super().__init__()
        self.kernel_len = kernel_len
        self.n_encoded = n_encoded
        self.conv1 = nn.Conv3d(4, n_encoded, 1, bias=bias)
        self.xconv1 = nn.Conv2d(n_encoded*kernel_len, n_encoded*kernel_len, 1, groups=kernel_len, bias=bias)
        self.yconv1 = nn.Conv2d(n_encoded*kernel_len, n_encoded*kernel_len, 1, groups=kernel_len, bias=bias)
        self.zconv1 = nn.Conv2d(n_encoded*kernel_len, n_encoded*kernel_len, 1, groups=kernel_len, bias=bias)
        self.conv3 = nn.Conv3d(n_encoded, n_encoded, 3, padding=1, bias=bias)
        # self.lin1 = Linear(5*n_encoded*kernel_len**3, 512, bias=bias)
        self.lin1 = Linear(n_encoded*kernel_len**3, 256, bias=bias)
        self.lin2 = Linear(256, 128, bias=bias)
        self.lin3 = Linear(128, 32, bias=bias)
        self.lin4 = Linear(32, 8, bias=bias)
        self.lin5 = Linear(8, 1, bias=bias)
        self.activation_shift = 0.0
        if activation == 'sig': 
            self.activation = Sigmoid()
            self.activation_shift = -0.5
        elif activation == 'elu': self.activation = ELU()
        else: raise NotImplementedError

    def forward(self, x):
        L = self.kernel_len
        C = self.n_encoded
        x = self.conv1(x)
        x_x = self.xconv1(x.view(-1, C*L, L, L)).view(-1, C, L, L, L)
        x_y = self.yconv1(x.transpose(2,3).reshape(-1, C*L, L, L)).view(-1, C, L, L, L)
        x_z = self.zconv1(x.transpose(2,4).reshape(-1, C*L, L, L)).view(-1, C, L, L, L)
        x_conv3 = self.conv3(x)
        # x = torch.cat((x, x_x, x_y, x_z, x_conv3), dim=1)
        x = x + x_x + x_y + x_z + x_conv3
        x = x.view(-1, self.n(x))
        print('1', x.shape, x.min().item(), x.max().item())
        x = self.activation(self.lin1(x)) + self.activation_shift
        print('2', x.min().item(), x.max().item())
        x = self.activation(self.lin2(x)) + self.activation_shift
        print('3', x.min().item(), x.max().item())
        x = self.activation(self.lin3(x)) + self.activation_shift
        print('4', x.min().item(), x.max().item())
        x = self.activation(self.lin4(x)) + self.activation_shift
        print('5', x.min().item(), x.max().item())
        x = self.lin5(x)
        print('6', x.min().item(), x.max().item())
        return x
    
    def n(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features