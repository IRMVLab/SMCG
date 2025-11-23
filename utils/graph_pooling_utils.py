import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# https://blog.csdn.net/qq_36643449/article/details/123529791


def normalize(A, symmetric=True):
    A = A + torch.eye(A.size(0))
    d = torch.abs(A.sum(1))


    if symmetric:
        D = torch.diag(torch.pow(d, -0.5))

        return D.mm(A).mm(D)  # D^(-1/2)AD^(-1/2)
    else:
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A)


class GCN_3_layers_(nn.Module):
    """
    Z = AXW
    """

    def __init__(self, dim_in, hidden_size1, hidden_size2, dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, hidden_size1)  # , bias=False)
        self.leaky_relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # , bias=False)
        self.leaky_relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.fc3 = nn.Linear(hidden_size2, dim_out)  # , bias=False)

    def forward(self, A, X):
        """
        Compute a three-layer GCN.
        """

        X = self.leaky_relu1(self.fc1(A.mm(X)))  # X corresponds to H in the equation
        X = self.leaky_relu2(self.fc2(A.mm(X)))

        return self.fc3(A.mm(X))


class GraphPooling_(nn.Module):
    """
    Z = AXWS
    """

    def __init__(self, dim_in, hidden_size1, hidden_size2, dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, hidden_size1)  # , bias=False)
        self.leaky_relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # , bias=False)
        self.leaky_relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.fc3 = nn.Linear(hidden_size2, dim_out)  # , bias=False)

    def forward(self, A, X, S):
        """
        Compute a three-layer GCN with pooling.
        """
        X = self.leaky_relu1(self.fc1(A.mm(X)))  # X corresponds to H in the equation
        X = self.leaky_relu2(self.fc2(A.mm(X)))
        return self.fc3(A.mm(X)).mm(S.transpose(1, 0))


