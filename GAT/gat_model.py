import torch
import torch.nn as nn
import torch.nn.functional as F
from GAT.layers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, device='cpu'):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.device = device

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True).to(self.device) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False).to(self.device)

    def forward(self, x, adj):
        # 开启梯度

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x = F.log_softmax(x, dim=0)

        linear_layer_1 = nn.Linear(x.size(0), 512).to(self.device)
        x = linear_layer_1(x.transpose(1, 0))
        x = F.leaky_relu(x, negative_slope=0.1)
        linear_layer_2 = nn.Linear(512, 256).to(self.device)
        x = linear_layer_2(x)
        # return F.log_softmax(x, dim=1)
        return x

# class SpGAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
#         """Sparse version of GAT."""
#         super(SpGAT, self).__init__()
#         self.dropout = dropout
#
#         self.attentions = [SpGraphAttentionLayer(nfeat,
#                                                  nhid,
#                                                  dropout=dropout,
#                                                  alpha=alpha,
#                                                  device=self.device,
#                                                  concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#
#         self.out_att = SpGraphAttentionLayer(nhid * nheads,
#                                              nclass,
#                                              dropout=dropout,
#                                              alpha=alpha,
#                                              concat=False,
#                                              device=self.device)
#
#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         return F.log_softmax(x, dim=1)
