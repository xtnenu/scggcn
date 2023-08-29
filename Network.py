import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import ChebConv,GCN,GAT,GATv2Conv
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops

class Gat(torch.nn.Module):
    def __init__(self, input_size,data):
        super(Gat, self).__init__()
        self.data=data
        self.conv1 = GATv2Conv(input_size, 300, num_layers=8)
        self.line = Linear(300, 1)

    def forward(self):
        edge_index = self.data.edge_index
        x = torch.relu(self.conv1(self.data.x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.line(x)
        x = torch.sigmoid(x)
        return x

class Gat_bayes(torch.nn.Module):
    def __init__(self,input_size,data):
        super(Gat_bayes, self).__init__()
        self.data=data
        self.conv1 = GATv2Conv(input_size,300,num_layers=4)
        self.conv2 = GATv2Conv(300, 100,num_layers=4)
        self.conv3 = GATv2Conv(100, 1, num_layers=1)
        self.lin1 = Linear(input_size, 100)
        self.lin2 = Linear(input_size, 100)
        self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self):
        edge_index, _ = dropout_adj(self.data.edge_index, p=0.5,
                                    force_undirected=True,
                                    num_nodes=self.data.x.size()[0],
                                    training=self.training)
        x0 = F.dropout(self.data.x, training=self.training)
        x = torch.relu(self.conv1(x0, edge_index))
        x = F.dropout(x, training=self.training)
        x1 = torch.relu(self.conv2(x, edge_index))
        x = x1 + torch.relu(self.lin1(x0))
        z = x1 + torch.relu(self.lin2(x0))
        pos_loss = -torch.log(torch.sigmoid((z[self.data.edge_index[0]] * z[self.data.edge_index[1]]).sum(dim=1)) + 1e-15).mean()
        pb, _ = remove_self_loops(self.data.edge_index)
        pb, _ = add_self_loops(pb)
        neg_edge_index = negative_sampling(pb, 13627, 504378)
        neg_loss = -torch.log(
            1 - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)) + 1e-15).mean()
        r_loss = pos_loss + neg_loss
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return x, r_loss, self.c1, self.c2

class ChebNet(torch.nn.Module):
    def __init__(self,input_size,data):
        super(ChebNet, self).__init__()
        self.data=data
        self.conv1 = ChebConv(input_size, 300, K=2, normalization="sym")
        self.conv2 = ChebConv(300, 100, K=2, normalization="sym")
        self.conv3 = ChebConv(100, 1, K=2, normalization="sym")
        self.lin1 = Linear(input_size, 100)
        self.lin2 = Linear(input_size, 100)
        self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self):
        edge_index, _ = dropout_adj(self.data.edge_index, p=0.5,
                                    force_undirected=True,
                                    num_nodes=self.data.x.size()[0],
                                    training=self.training)
        x0 = F.dropout(self.data.x, training=self.training)
        x = torch.relu(self.conv1(x0, edge_index))
        x = F.dropout(x, training=self.training)
        x1 = torch.relu(self.conv2(x, edge_index))
        x = x1 + torch.relu(self.lin1(x0))
        z = x1 + torch.relu(self.lin2(x0))
        pos_loss = -torch.log(torch.sigmoid((z[self.data.edge_index[0]] * z[self.data.edge_index[1]]).sum(dim=1)) + 1e-15).mean()
        pb, _ = remove_self_loops(self.data.edge_index)
        pb, _ = add_self_loops(pb)
        neg_edge_index = negative_sampling(pb, 13627, 504378)
        neg_loss = -torch.log(
            1 - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)) + 1e-15).mean()
        r_loss = pos_loss + neg_loss
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return x, r_loss, self.c1, self.c2

class Gcn(torch.nn.Module):
    def __init__(self, input_size,data):
        super(Gcn, self).__init__()
        self.data=data
        self.conv1 = GCN(input_size, 300, num_layers=4)
        self.conv2 = GCN(300, 100, num_layers=4)
        self.line = Linear(100, 1)

    def forward(self):
        edge_index = self.data.edge_index
        x = torch.relu(self.conv1(self.data.x, edge_index))
        x = F.dropout(x, training=self.training)
        x = torch.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.line(x)
        x = torch.sigmoid(x)
        return x

class Gcn_bayes(torch.nn.Module):
    def __init__(self,input_size,data):
        super(Gcn_bayes, self).__init__()
        self.conv1 = GCN(input_size,300,num_layers=4)
        self.conv2 = GCN(300, 100,num_layers=4)
        self.conv3 = GCN(100, 1, num_layers=1)
        self.lin1 = Linear(input_size, 100)
        self.lin2 = Linear(input_size, 100)
        self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self):
        edge_index, _ = dropout_adj(self.data.edge_index, p=0.5,
                                    force_undirected=True,
                                    num_nodes=self.data.x.size()[0],
                                    training=self.training)
        x0 = F.dropout(self.data.x, training=self.training)
        x = torch.relu(self.conv1(x0, edge_index))
        x = F.dropout(x, training=self.training)
        x1 = torch.relu(self.conv2(x, edge_index))
        x = x1 + torch.relu(self.lin1(x0))
        z = x1 + torch.relu(self.lin2(x0))
        pos_loss = -torch.log(torch.sigmoid((z[self.data.edge_index[0]] * z[self.data.edge_index[1]]).sum(dim=1)) + 1e-15).mean()
        pb, _ = remove_self_loops(self.data.edge_index)
        pb, _ = add_self_loops(pb)
        neg_edge_index = negative_sampling(pb, 13627, 504378)
        neg_loss = -torch.log(
            1 - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)) + 1e-15).mean()
        r_loss = pos_loss + neg_loss
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return x, r_loss, self.c1, self.c2