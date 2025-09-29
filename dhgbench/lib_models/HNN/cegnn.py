import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv

class CEGCN(nn.Module):
    def __init__(self,num_features, num_targets, args):
        super(CEGCN, self).__init__()
        
        self.hidden_dim = args.MLP_hidden
        self.num_layers = args.All_num_layers
        self.dropout=args.dropout
        self.use_bn=args.use_bn

        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList()

        if self.num_layers == 1:
            self.convs.append(GCNConv(in_channels=num_features, out_channels=num_targets))
        else:
            self.convs.append(GCNConv(in_channels=num_features, out_channels=self.hidden_dim))
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(self.hidden_dim))
            for l in range(self.num_layers - 2):
                self.convs.append(GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim))
                if self.use_bn:
                    self.bns.append(nn.BatchNorm1d(self.hidden_dim))
            self.convs.append(GCNConv(in_channels=self.hidden_dim, out_channels=num_targets))

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()

        if self.use_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.clique_edge_index
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x, None


class CEGAT(nn.Module):
    def __init__(self,num_features, num_targets, args):
        super(CEGAT, self).__init__()
        
        self.hidden_dim = args.MLP_hidden
        self.num_layers = args.All_num_layers
        self.dropout=args.dropout

        self.heads = args.heads
        self.use_bn = args.use_bn
        self.concat = args.concat

        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList()

        if self.num_layers == 1:
            self.convs.append(GATConv(in_channels=num_features, 
                                      out_channels=num_targets,
                                      heads=1,
                                      concat=False)) 
        else:
            self.convs.append(GATConv(in_channels=num_features, 
                                      out_channels=self.hidden_dim,
                                      heads=self.heads,
                                      concat=self.concat))
            if self.use_bn:
                dim1 = self.hidden_dim * self.heads if self.concat else self.hidden_dim
                self.bns.append(nn.BatchNorm1d(dim1))
                        
            for l in range(self.num_layers - 2):
                in_dim = self.hidden_dim * self.heads if self.concat else self.hidden_dim
                self.convs.append(GATConv(in_channels=in_dim, 
                                          out_channels=self.hidden_dim,
                                          heads=self.heads,
                                          concat=self.concat))
                if self.use_bn:
                    out_dim = self.hidden_dim * self.heads if self.concat else self.hidden_dim
                    self.bns.append(nn.BatchNorm1d(out_dim))
            
            final_in = self.hidden_dim * self.heads if self.concat else self.hidden_dim
            self.convs.append(GATConv(in_channels=final_in, 
                                      out_channels=num_targets,
                                      heads=1,
                                      concat=False))

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
        if self.use_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.clique_edge_index
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x, None