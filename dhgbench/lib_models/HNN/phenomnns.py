import torch.nn as nn
import torch.nn.functional as F
from lib_models.HNN.mlp import MLP

class PhenomNNSMultiConv(nn.Module):

    def __init__(self, args):
        super(PhenomNNSMultiConv, self).__init__()
        self.args = args
        self.lam0=args.lam0
        self.lam1=args.lam1
        self.alpha=args.alpha if args.alpha !=0 else 1/(1+args.lam0+args.lam1)
        self.num_steps=args.prop_step 
    
    def forward(self, X, A, G):
        I,D=G[1],G[0]

        Y = Y0 = X
        Q_tild=D + I

        for k in range(self.num_steps):
            Y_hat = A @ Y + Y0 
            Y = F.relu((1 - self.alpha) * Y + self.alpha * (Q_tild ** -1) @ Y_hat)

        return Y

class PhenomNNS(nn.Module):
    def __init__(self, num_features, num_targets, args):
        super(PhenomNNS, self).__init__()
        
        self.hidden_dim = args.MLP_hidden 
        self.norm_type=args.normalization
        self.dropout = args.dropout
        self.act_fn = nn.ReLU()

        self.projector = MLP(
            in_channels=num_features,
            hidden_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            num_layers=args.encoder_num_layers,
            dropout=self.dropout,
            Normalization=self.norm_type,
            InputNorm=False,
        )

        self.decoder = MLP(
            in_channels=self.hidden_dim,
            hidden_channels=args.decoder_hidden,
            out_channels=num_targets,
            num_layers=args.decoder_num_layers,
            dropout=self.dropout,
            Normalization=self.norm_type,
            InputNorm=False,
        )
        
        self.conv = PhenomNNSMultiConv(args=args)

    def reset_parameters(self):

        self.projector.reset_parameters()
        self.decoder.reset_parameters()
        
    def forward(self, data):

        input,adj,G = data.x, data.adj, data.G

        x = F.dropout(input, self.dropout, training=self.training)
        x = F.relu(self.projector(x))
        x = self.conv(x, A=adj, G=G) 
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.decoder(x)
        return x,None