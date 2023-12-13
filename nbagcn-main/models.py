import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc11 = GraphConvolution(nhid1, nhid2)
        # self.gc111 = GraphConvolution(nhid2, nhid3)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    # def forward(self, x, adj):
    #     x = F.relu(self.gc1(x, adj))
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = self.gc2(x, adj)
    #     return F.log_softmax(x, dim=1)
    
    def forward(self, x, adj):
        x = F.tanh(self.gc1(x, adj))
        # x = F.relu(self.gc11(x, adj))
        # x = F.relu(self.gc111(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
