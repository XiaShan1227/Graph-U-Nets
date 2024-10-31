#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/10/27 16:12
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_dense_adj, dense_to_sparse


class GPool(nn.Module):
    def __init__(self, num_f, pool_ratio):
        super(GPool, self).__init__()

        self.p = nn.Linear(num_f, 1, bias=False)
        self.ratio = pool_ratio # 池化率

    def forward(self, xl, edge_index):
        y = self.p(xl) / torch.norm(self.p.weight) # 投影向量

        k = int(self.ratio * len(y))
        topk_values, top_idxs = torch.topk(y, k, dim=0) # 获取投影向量前k个最大值以及索引

        y_hat = torch.sigmoid(topk_values)
        xl_hat = xl[top_idxs, :].squeeze()
        xl1 = xl_hat * y_hat

        al = to_dense_adj(edge_index, max_num_nodes=len(y)).squeeze() # l层邻接矩阵(稠密形)

        al1 = torch.index_select(al, 0, top_idxs.squeeze())
        al1 = torch.index_select(al1, 1, top_idxs.squeeze()) # l+1层邻接矩阵(稠密形)

        al1_sparse = dense_to_sparse(al1) # l+1层邻接矩阵(稀疏形)
        edge_index_pool = torch.sparse_coo_tensor(al1_sparse[0], al1_sparse[1]).coalesce().indices()  # 可以去除重复边

        return xl1, edge_index_pool, top_idxs.squeeze()


""" 对于batch-graph可以使用此代码  
class GPool(nn.Module):
    def __init__(self, num_f, pool_ratio):
        super(GPool, self).__init__()

        self.p = nn.Linear(num_f, 1, bias=False)
        self.ratio = pool_ratio # 池化率

    def forward(self, xl, edge_index, batch):
        y = self.p(xl) / torch.norm(self.p.weight) # 投影向量
        unique_graphs = batch.unique()  # 获取所有图的唯一标识符

        top_idxs = []
        for graph_id in unique_graphs:
            # 获取属于该图的节点索引
            node_idxs = (batch == graph_id).nonzero(as_tuple=True)[0]
            # 对该图的投影向量计算 topk
            y_graph = y[node_idxs]
            k = max(1, math.ceil(self.ratio * len(y_graph)))  # 至少保留一个节点
            _, topk_idxs = torch.topk(y_graph, k, dim=0)
            # 映射回全局索引
            top_idxs.append(node_idxs[topk_idxs.squeeze()])

        top_idxs = torch.cat(top_idxs, dim=0)  # 所有图的保留节点索引
        y_hat = torch.sigmoid(y[top_idxs])
        xl1 = xl[top_idxs] * y_hat

        al = to_dense_adj(edge_index, max_num_nodes=len(y)).squeeze() # l层邻接矩阵(稠密形)

        al1 = torch.index_select(al, 0, top_idxs.squeeze())
        al1 = torch.index_select(al1, 1, top_idxs.squeeze()) # l+1层邻接矩阵(稠密形)

        al1_sparse = dense_to_sparse(al1) # l+1层邻接矩阵(稀疏形)
        edge_index_pool = torch.sparse_coo_tensor(al1_sparse[0], al1_sparse[1]).coalesce().indices()  # 可以去除重复边

        batch_pool = batch[top_idxs] # 更新池化后的batch信息

        return xl1, edge_index_pool, top_idxs.squeeze(), batch_pool
"""


class GUnpool(nn.Module):
    def __init__(self):
        super(GUnpool, self).__init__()

    def forward(self, xl, idxs, up_shape):
        xl1 = torch.zeros(up_shape[0], xl.shape[1]).to(xl.device) # 图中节点特征初始化为0
        xl1[idxs] = xl # 反池化前的图特征

        return xl1


class GraphUNets(nn.Module):
    def __init__(self, args, num_features, num_classes):
        super(GraphUNets, self).__init__()

        self.conv1 = GCNConv(num_features, 32, improved=args.improved)
        self.conv2 = GCNConv(32, 64, improved=args.improved)
        self.conv3 = GCNConv(64, 128, improved=args.improved)
        self.conv4 = GCNConv(128, 256, improved=args.improved)

        self.pool1 = GPool(32, args.pooling_ratio)
        self.pool2 = GPool(64, args.pooling_ratio)
        self.pool3 = GPool(128, args.pooling_ratio)

        self.unpool = GUnpool()

        self.conv5 = GCNConv(256 + 128, 128, improved=args.improved)
        self.conv6 = GCNConv(128 + 64, 64, improved=args.improved)
        self.conv7 = GCNConv(64 + 32, 32, improved=args.improved)

        self.ac = nn.ELU(alpha=1.0)

        self.l1 = nn.Linear(64, 64, bias=False)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, data):
        """
        data.x: 一批图的节点特征 [batch_size*num_nodes, num_features] ——> [bs*num_n, num_f]
        data.edge_index: 一批图的邻接矩阵 [2, num_edges]
        data.batch: 确保每张图上节点映射到同一batch [0,0,...5,5,...batch_size-1,batch_size-1]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        ## Encoder
        # conv1
        x1 = self.conv1(x, edge_index) # [bs*num_n1, num_f] ——> [bs*num_n1, num_f=32]
        x1 = self.ac(x1)
        # pool1
        x2, edge_index2, idx2 = self.pool1(x1, edge_index) # [bs*num_n1, num_f=32] ——> [bs*num_n2, num_f=32]
        x2 = self.ac(x2)
        # conv2
        x3 = self.conv2(x2, edge_index2) # [bs*num_n2, num_f=32] ——> [bs*num_n2, num_f=64]
        x3 = self.ac(x3)
        # pool2
        x4, edge_index4, idx4 = self.pool2(x3, edge_index2)  # [bs*num_n2, num_f=64] ——> [bs*num_n3, num_f=64]
        x4 = self.ac(x4)
        # conv3
        x5 = self.conv3(x4, edge_index4) # [bs*num_n3, num_f=64] ——> [bs*num_n3, num_f=128]
        x5 = self.ac(x5)
        # pool3
        x6, edge_index6, idx6 = self.pool3(x5, edge_index4)  # [bs*num_n3, num_f=128] ——> [bs*num_n4, num_f=128]
        x6 = self.ac(x6)
        # conv4
        x7 = self.conv4(x6, edge_index6) # [bs*num_n4, num_f=128] ——> [bs*num_n4, num_f=256]
        x7 = self.ac(x7)

        ## Decoder
        # unpool1
        x8 = self.unpool(x7, idx6, x5.shape) # [bs*num_n4, num_f=256] ——> [bs*num_n3, num_f=256]
        x8 = torch.cat([x8, x5], dim=1) # [bs*num_n3, num_f=256+128]
        x8 = self.ac(x8)
        # conv5
        x9 = self.conv5(x8, edge_index4) # [bs*num_n3, num_f=256+128] ——> [bs*num_n3, num_f=128]
        x9 = self.ac(x9)
        # unpool2
        x10 = self.unpool(x9, idx4, x3.shape) # [bs*num_n3, num_f=128] ——> [bs*num_n2, num_f=128]
        x10 = torch.cat([x10, x3], dim=1) # [bs*num_n2, num_f=128+64]
        x10 = self.ac(x10)
        # conv6
        x11 = self.conv6(x10, edge_index2) # [bs*num_n2, num_f=128+64] ——> [bs*num_n2, num_f=64]
        x11 = self.ac(x11)
        # unpool3
        x12 = self.unpool(x11, idx2, x1.shape) # [bs*num_n2, num_f=64] ——> [bs*num_n1, num_f=64]
        x12 = torch.cat([x12, x1], dim=1) # [bs*num_n1, num_f=64+32]
        x12 = self.ac(x12)
        # conv7
        x13 = self.conv7(x12, edge_index)  # [bs*num_n1, num_f=64+32] ——> [bs*num_n1, num_f=32]
        x13 = self.ac(x13)


        ## Readout
        x = torch.cat([gmp(x13, batch), gap(x13, batch)], dim=1) # [bs*num_n1, num_f=32] ——> [bs, num_f=32*2=64]
        x = self.ac(x)

        x = self.l1(x) # [bs, num_f=64] ——> [bs, num_f=64]
        x = self.ac(x)

        x = self.classifier(x)  # (bs, num_f=64) ——> (bs, datasets_number_categories)

        x = F.log_softmax(x, dim=-1)

        return x
