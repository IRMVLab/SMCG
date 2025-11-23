import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.pool import global_max_pool


class GCN_2_layers(nn.Module):
    def __init__(self, in_feats, hidden_size, out):
        super(GCN_2_layers, self).__init__()
        self.conv1 = GCNConv(in_feats, hidden_size)
        self.conv2 = GCNConv(hidden_size, out)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x, edge_index)
        h = torch.relu(h)
        h = self.conv2(h, edge_index)
        return h


class GCN_3_layers(nn.Module):
    def __init__(self, in_feats, hidden_size1, hidden_size2, out):
        super(GCN_3_layers, self).__init__()
        self.conv1 = GCNConv(in_feats, hidden_size1)
        self.leaky_relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = GCNConv(hidden_size1, hidden_size2)
        self.leaky_relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv3 = GCNConv(hidden_size2, out)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x, edge_index)
        h = self.leaky_relu1(h)
        h = self.conv2(h, edge_index)
        h = self.leaky_relu2(h)
        h = self.conv3(h, edge_index)
        return h


class GraphPooling(nn.Module):
    """
    Z = AXWS
    """

    def __init__(self, in_feats, hidden_size1, hidden_size2, out):
        super().__init__()
        self.conv1 = GATConv(in_feats, hidden_size1)
        self.leaky_relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = GATConv(hidden_size1, hidden_size2)
        self.leaky_relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv3 = GATConv(hidden_size2, out)

    def forward(self, data, S):
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x, edge_index)
        h = self.leaky_relu1(h)
        h = self.conv2(h, edge_index)
        h = self.leaky_relu2(h)
        h = self.conv3(h, edge_index)
        return h @ S.t()
        # x_pooled = global_max_pool(h, batch=)  # Max pooling operation
        #
        # return x_pooled
