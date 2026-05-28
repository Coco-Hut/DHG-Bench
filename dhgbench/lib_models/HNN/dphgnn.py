import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_scatter import scatter_softmax, scatter_sum, scatter_mean, scatter_add
from lib_models.HNN.utils import SparseMM

class DPHGNN(nn.Module):
    def __init__(self, num_features, num_targets, args):
        super(DPHGNN, self).__init__()
        
        self.fc_dim = args.fc_dim
        self.dff_MLP_hidden = args.dff_MLP_hidden
        
        self.taa_layer = TAAMSG(num_features,args)
        self.spectral_layer = HGSpectralNet(num_features,args.spectral_embed_dim)
        self.fc = nn.Linear(in_features=self.spectral_layer.out_channels+self.taa_layer.out_channels, 
                            out_features=self.fc_dim).to(args.device)
        if getattr(args, 'DPHGNN_dff_mode', 'attention') == 'paper':
            self.dff_layer = PaperDFF(self.fc_dim,num_targets,args)
        else:
            self.dff_layer = DFF(self.fc_dim,num_targets,args)
    
    def reset_parameters(self):
        self.taa_layer.reset_parameters()
        self.spectral_layer.reset_parameters()
        self.fc.reset_parameters()
        self.dff_layer.reset_parameters()
        
    def forward(self,data):
        
        # 1. TAA message passing
        taa_features,_, S_star = self.taa_layer(data)
        
        # 2. feature mixture module
        spectral_features = self.spectral_layer(data)
        
        X = torch.cat([taa_features, spectral_features], dim=1) 
        X = self.fc(X)
        
        # 3. Dynamic Feature Fusion
        hyperedge_index = data.hyperedge_index
        X = self.dff_layer(X,hyperedge_index,S_star)
        
        return X,None

#--------------------- Dynamic Feature Fusion Module ---------------------

class DFF(nn.Module):
    def __init__(self, in_channels:int,out_channels: int, args):
        super(DFF, self).__init__()
        
        self.dff_MLP_hidden=args.dff_MLP_hidden 
        self.dff_num_layers = args.dff_num_layers
        
        self.convs = nn.ModuleList()
        self.convs.append(DPHGNNConv(in_channels, self.dff_MLP_hidden,args,is_last=False))
        for _ in range(self.dff_num_layers-1):
            self.convs.append(DPHGNNConv(self.dff_MLP_hidden, self.dff_MLP_hidden,args,is_last=False))
        self.convs.append(HypergraphConv(self.dff_MLP_hidden, out_channels, is_last=True))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, X, HG, S_features):
        for conv in self.convs:
            X = conv(X, HG, S_features)
        return X

class HypergraphConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels, 
        bias: bool = True,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.dropout=drop_rate
        self.W = nn.Linear(in_channels, out_channels, bias=bias)

    def reset_parameters(self):
        self.W.reset_parameters()
    
    def forward(self, X, hyperedge_index, S_features):
        
        V, E = hyperedge_index
        num_nodes,num_edges = V.max()+1,E.max()+1
        
        X = self.W(X)
        
        # Y = hg.v2e(X, aggr="mean")
        Xve = X[V]  # Shape: (num_edges, feature_dim)
        Y = scatter_mean(Xve, E, dim=0, dim_size=num_edges)  # Shape: (num_edges, feature_dim)
        
        ones = torch.ones(V.shape[0], device=V.device, dtype=torch.float32)
        D_v = torch_scatter.scatter_add(ones, V, dim=0)
        
        _De = torch.zeros(num_edges, device=V.device)
        _De = _De.scatter_reduce(0, index=E, src=D_v[V], reduce="mean")
        _De = _De.pow(-0.5)
        _De[_De.isinf()] = 1
        Y = _De.view(-1, 1) * Y
        
        Xev = Y[E]
        X = scatter_add(Xev, V, dim=0, dim_size=num_nodes)
        _Dv = D_v.pow(-0.5)
        _Dv[_Dv.isinf()] = 0
        X = _Dv.view(-1, 1) * X
        
        if not self.is_last:
            X = F.relu(X)
            X = F.dropout(X, self.dropout, training=self.training)
            
        return X

class PaperDFF(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, args):
        super(PaperDFF, self).__init__()
        self.dff_MLP_hidden = args.dff_MLP_hidden
        self.dff_num_layers = args.dff_num_layers

        self.convs = nn.ModuleList()
        self.convs.append(PaperDPHGNNConv(in_channels, self.dff_MLP_hidden, args, is_last=False))
        for _ in range(self.dff_num_layers - 1):
            self.convs.append(PaperDPHGNNConv(self.dff_MLP_hidden, self.dff_MLP_hidden, args, is_last=False))
        self.convs.append(PaperHypergraphConv(self.dff_MLP_hidden, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, X, HG, S_features):
        for conv in self.convs[:-1]:
            X = conv(X, HG, S_features)
        return self.convs[-1](X, HG, S_features)

class PaperDPHGNNConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        args,
        bias: bool = True,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super(PaperDPHGNNConv, self).__init__()
        self.is_last = is_last
        self.dropout = drop_rate
        self.star_proj = nn.Linear(args.expan_dim, in_channels, bias=bias)
        self.msg_proj = nn.Linear(in_channels, out_channels, bias=bias)
        self.skip_proj = nn.Linear(in_channels, out_channels, bias=bias)

    def reset_parameters(self):
        self.star_proj.reset_parameters()
        self.msg_proj.reset_parameters()
        self.skip_proj.reset_parameters()

    def forward(self, X, hyperedge_index, S_features):
        V, E = hyperedge_index
        num_nodes = X.size(0)
        num_edges = S_features.size(0)

        ones = torch.ones(V.shape[0], device=V.device, dtype=X.dtype)
        D_v = torch_scatter.scatter_add(ones, V, dim=0, dim_size=num_nodes).clamp_min(1)
        D_e = torch_scatter.scatter_add(ones, E, dim=0, dim_size=num_edges).clamp_min(1)
        Dv_inv_sqrt = D_v.pow(-0.5)
        De_inv = D_e.pow(-1.0)

        node_to_edge = scatter_add(X[V] * Dv_inv_sqrt[V].unsqueeze(-1), E, dim=0, dim_size=num_edges)
        star_term = self.star_proj(S_features) * De_inv.unsqueeze(-1)
        fused_edge = node_to_edge + star_term

        edge_to_node = scatter_add(fused_edge[E] * De_inv[E].unsqueeze(-1), V, dim=0, dim_size=num_nodes)
        X = self.skip_proj(X) + self.msg_proj(edge_to_node)

        if not self.is_last:
            X = F.elu(X)
            X = F.dropout(X, self.dropout, training=self.training)

        return X

class PaperHypergraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias: bool = True):
        super(PaperHypergraphConv, self).__init__()
        self.W = nn.Linear(in_channels, out_channels, bias=bias)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, X, hyperedge_index, S_features):
        V, E = hyperedge_index
        num_nodes = X.size(0)
        num_edges = int(E.max().item()) + 1

        ones = torch.ones(V.shape[0], device=V.device, dtype=X.dtype)
        D_v = torch_scatter.scatter_add(ones, V, dim=0, dim_size=num_nodes).clamp_min(1)
        D_e = torch_scatter.scatter_add(ones, E, dim=0, dim_size=num_edges).clamp_min(1)
        Dv_inv_sqrt = D_v.pow(-0.5)
        De_inv = D_e.pow(-1.0)

        node_to_edge = scatter_add(X[V] * Dv_inv_sqrt[V].unsqueeze(-1), E, dim=0, dim_size=num_edges)
        edge_to_node = scatter_add(node_to_edge[E] * De_inv[E].unsqueeze(-1), V, dim=0, dim_size=num_nodes)
        return self.W(edge_to_node)

class DPHGNNConv(nn.Module):
    def __init__(self,
               in_channels,
               out_channels,
               args,
               bias: bool=True,
               drop_rate: float=0.5,
               is_last: bool=False
               ):
        super(DPHGNNConv, self).__init__()
        
        self.is_last = is_last
        self.star_dim = args.expan_dim
        
        self.W_x = nn.Linear(in_channels, out_channels, bias=bias)
        self.W_vertex = nn.Linear(in_channels, out_channels, bias=bias)
        self.atten_vertex = nn.Linear(out_channels, 1, bias=False) 
        
        self.theta_e2v = nn.Linear(out_channels+self.star_dim, out_channels, bias = bias)
        
        self.atten_act = nn.LeakyReLU(args.atten_neg_slope)
        self.dropout=drop_rate
    
    def reset_parameters(self):
        self.W_x.reset_parameters()
        self.W_vertex.reset_parameters()
        self.atten_vertex.reset_parameters()
        self.theta_e2v.reset_parameters()
    
    def forward(self, X, hyperedge_index, S_features):
        
        V, E = hyperedge_index
        num_nodes,num_edges = V.max()+1,E.max()+1
        
        X_init = self.W_x(X) 
        
        X_feat = self.W_vertex(X) # [|V|,out_channels]
        x_for_vertex = self.atten_vertex(X_feat) # [|V|,1]
        
        v2e_atten_score = x_for_vertex[hyperedge_index[0]]
        v2e_atten_score = F.dropout(self.atten_act(v2e_atten_score).squeeze(),self.dropout, training=self.training)

        softmax_weight = scatter_softmax(v2e_atten_score, E, dim=0)  # (nnz,)

        Xev = X_feat[V] * softmax_weight.unsqueeze(-1)  # (nnz, d)
        Y_v2e = scatter_sum(Xev, E, dim=0, dim_size=num_edges)  # (num_edges, d)
        Y_v2e = F.elu(Y_v2e)
        
        e_msg = torch.cat((Y_v2e, S_features), dim=1) 
        Y = self.theta_e2v(e_msg)
        
        Yev = Y[E]  # (nnz, d)

        X = scatter_mean(Yev, V, dim=0, dim_size=num_nodes)  # (num_nodes, d)
        
        if not self.is_last:
            X = F.elu(X)
            X = F.dropout(X, self.dropout, training=self.training)
            
        return X + X_init 
    
#--------------------- Topology-Aware Attention Layer ---------------------      
    
class TAAMSG(nn.Module):
    def __init__(self, num_features, args):
        super(TAAMSG, self).__init__()
        
        self.expan_dim = args.expan_dim
        self.spatial_feat_dim =args.taa_spatial_dim
        self.spectral_feat_dim = args.taa_spectral_dim
        self.num_heads =args.num_heads 
        self.chunk_size = args.chunk_size 
        self.out_channels = self.spatial_feat_dim+self.spectral_feat_dim 
        
        self.cg_layer = ExpanNet(num_features,self.expan_dim,args,var='CG')
        self.hg_layer = ExpanNet(num_features,self.expan_dim,args,var='HG')
        self.sg_layer = ExpanNet(num_features,self.expan_dim,args,var='SG')
        self.taa_spatial_layer = TAA_Spatial(self.expan_dim,self.spatial_feat_dim,self.num_heads,self.chunk_size)
        self.spectral_net = SmoothNet(self.expan_dim,self.expan_dim)
        self.taa_spectral_layer = TAA_Spectral(self.expan_dim,self.spectral_feat_dim,self.num_heads,self.chunk_size)
    
    def reset_parameters(self):
        self.cg_layer.reset_parameters()
        self.hg_layer.reset_parameters()
        self.sg_layer.reset_parameters()
        self.taa_spatial_layer.reset_parameters()
        self.spectral_net.reset_parameters()
        self.taa_spectral_layer.reset_parameters()
    
    def get_star_features(self,data):
        x = data.x
        num_hyperedges = data.hyperedge_index[1].max()+1
        s_init = torch.zeros(num_hyperedges, data.num_features).to(x.device)
        x_star = torch.cat((x, s_init), 0)
        return x_star
    
    def forward(self,data):
        
        '''
        1. Expansion Graph Convolution
        '''
        x_star = self.get_star_features(data) 
        clique_graph_features = self.cg_layer(data.x,data.A_cg)
        hypergcn_graph_features = self.hg_layer(data.x,data.A_hyp) 
        star_graph_features = self.sg_layer(x_star,data.A_sg) # 10->64, [94318, 64]

        X_star, S_star = star_graph_features[data.v_mask], star_graph_features[~data.v_mask] # X_star: [66790, 64], S_star: [27528, 64]
        
        '''
        2. Feature Mixture Module
        '''
        X_hyp, X_cg, X_sg = self.spectral_net(hypergcn_graph_features,clique_graph_features,star_graph_features,data.L_hyp,data.L_cg,data.L_sg)
        X_spectral = self.taa_spectral_layer(X_hyp, X_cg, X_sg[data.v_mask]) # [66790, 10]
        
        # return hypergcn_graph_features, clique_graph_features, X_star
        
        X_spatial = self.taa_spatial_layer(hypergcn_graph_features, clique_graph_features, X_star) # [66790, 64]
        
        taa_features = torch.cat((X_spatial, X_spectral), 1) # [66790, 76]
        
        return taa_features, X_star, S_star    
    
class TAA_Spectral(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, chunk_size):
        super(TAA_Spectral, self).__init__()
        
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        
        self.hyp_proj = nn.Linear(in_channels, out_channels)
        self.star_proj = nn.Linear(in_channels, out_channels)
        self.clique_proj = nn.Linear(in_channels, out_channels)
        self.MHA = nn.MultiheadAttention(out_channels, num_heads)

    def chunked_MHA(self, Q, K, V, chunk_size):
        num_nodes = Q.shape[0]
        outputs = []
        for i in range(0, num_nodes, chunk_size):
            Q_chunk = Q[i : i + chunk_size]
            K_chunk = K[i : i + chunk_size]
            V_chunk = V[i : i + chunk_size]
            attn_out, _ = self.MHA(Q_chunk, K_chunk, V_chunk)
            outputs.append(attn_out)
        return torch.cat(outputs, dim=0)

    def reset_parameters(self):
        
        self.hyp_proj.reset_parameters()
        self.star_proj.reset_parameters()
        self.clique_proj.reset_parameters()

        nn.init.xavier_uniform_(self.MHA.in_proj_weight)
        if self.MHA.in_proj_bias is not None:
            nn.init.constant_(self.MHA.in_proj_bias, 0.)

        nn.init.xavier_uniform_(self.MHA.out_proj.weight)
        if self.MHA.out_proj.bias is not None:
            nn.init.constant_(self.MHA.out_proj.bias, 0.)

    def forward(self, X_L_hyp, X_L_star, X_L_clique):
        
        hyp_proj = self.hyp_proj(X_L_hyp)
        star_proj = self.star_proj(X_L_star)
        clique_proj = self.clique_proj(X_L_clique)
        if self.chunk_size==-1:
            spectral_attn_output, _ = self.MHA(hyp_proj, star_proj, clique_proj)
        else:
            spectral_attn_output = self.chunked_MHA(hyp_proj, star_proj, clique_proj,self.chunk_size)
        
        return spectral_attn_output  

class SmoothNet(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(SmoothNet, self).__init__()
        self.W_hyp = nn.Linear(in_channels, out_channels, bias = bias)
        self.W_cg = nn.Linear(in_channels, out_channels, bias = bias)
        self.W_sg = nn.Linear(in_channels, out_channels, bias = bias)

    def reset_parameters(self):
        self.W_hyp.reset_parameters()
        self.W_cg.reset_parameters()
        self.W_sg.reset_parameters()
    
    def forward(self, X_HypGNet, X_CGNet, X_SGNet, L_hyp, L_cg, L_sg):
        
        X_hyp = self.W_hyp(X_HypGNet)
        X_cg = self.W_cg(X_CGNet)
        X_sg = self.W_sg(X_SGNet)
        
        X_hyp_s = SparseMM.apply(L_hyp, X_hyp)
        X_cg_s = SparseMM.apply(L_cg, X_cg)
        X_sg_s = SparseMM.apply(L_sg, X_sg)
        
        return X_hyp_s, X_cg_s, X_sg_s

class TAA_Spatial(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, chunk_size):
        super(TAA_Spatial, self).__init__()
        
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        
        self.hyp_proj = nn.Linear(in_channels, out_channels)
        self.star_proj = nn.Linear(in_channels, out_channels)
        self.clique_proj = nn.Linear(in_channels, out_channels)
        self.MHA = nn.MultiheadAttention(out_channels, num_heads)

    def chunked_MHA(self, Q, K, V, chunk_size):
        num_nodes = Q.shape[0]
        outputs = []
        for i in range(0, num_nodes, chunk_size):
            Q_chunk = Q[i : i + chunk_size]
            K_chunk = K[i : i + chunk_size]
            V_chunk = V[i : i + chunk_size]
            attn_out, _ = self.MHA(Q_chunk, K_chunk, V_chunk)
            outputs.append(attn_out)
        return torch.cat(outputs, dim=0)
    
    def reset_parameters(self):
        
        self.hyp_proj.reset_parameters()
        self.star_proj.reset_parameters()
        self.clique_proj.reset_parameters()

        nn.init.xavier_uniform_(self.MHA.in_proj_weight)
        if self.MHA.in_proj_bias is not None:
            nn.init.constant_(self.MHA.in_proj_bias, 0.)

        nn.init.xavier_uniform_(self.MHA.out_proj.weight)
        if self.MHA.out_proj.bias is not None:
            nn.init.constant_(self.MHA.out_proj.bias, 0.)
    
    def forward(self, X_hyp, X_star, X_clique):
        
        hyp_proj = self.hyp_proj(X_hyp)
        star_proj = self.star_proj(X_star)
        clique_proj = self.clique_proj(X_clique)
        
        if self.chunk_size==-1:
            spatial_attn_output, _ = self.MHA(hyp_proj, star_proj, clique_proj)
        else:
            spatial_attn_output = self.chunked_MHA(hyp_proj, star_proj, clique_proj,self.chunk_size)
        
        return spatial_attn_output

class HGSpectralNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias=True, drop_rate: float=0.3):
        super(HGSpectralNet, self).__init__()
        self.dropout = drop_rate
        self.out_channels = out_channels
        self.W = nn.Linear(in_channels*2, out_channels, bias= bias)

    def reset_parameters(self):
        self.W.reset_parameters()
    
    def forward(self, data):
        
        x, L_hgnn, L_sym, L_rw = data.x,data.L_hgnn,data.L_sym,data.L_rw
        
        X_hgnn = SparseMM.apply(L_hgnn, x)
        X_sym = SparseMM.apply(L_sym, x)
        X_rw = SparseMM.apply(L_rw, x)
        
        X_sym_rw = (X_sym + X_rw)/2
        X_spectral = torch.cat([X_hgnn, X_sym_rw], dim=1)
        
        X = F.relu(self.W(X_spectral))
        X = F.dropout(X, self.dropout, training=self.training)

        return X

#--------------------- Convolution for Expansion Graph ---------------------

class ExpanNet(nn.Module):
    def __init__(self, in_channels, out_channels, args, var = 'CG'):
        super(ExpanNet, self).__init__()

        # var choices = ['CG','HG','SG']
        self.args =args
        self.num_layers = eval(f'args.{var.lower()}_num_layer')
        self.hidden_dim =eval(f'args.{var.lower()}_MLP_hidden')
        self.dropout = eval(f'args.{var.lower()}_dropout') 

        self.convs = nn.ModuleList()
        if self.num_layers == 1:
            self.convs.append(GCNConv(in_channels,out_channels))
        else:
            self.convs.append(GCNConv(in_channels,self.hidden_dim))
            for _ in range(self.num_layers-2):
                self.convs.append(GCNConv(self.hidden_dim,self.hidden_dim))
            # Output heads is set to 1 as default
            self.convs.append(GCNConv(
                self.hidden_dim, out_channels))

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
            
    def forward(self,x,A):

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x,A)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, A)
        
        return x

class GCNConv(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_features, adj):
        support = SparseMM.apply(input_features, self.weight)
        output = SparseMM.apply(adj, support)

        if self.bias is not None:
            output = output + self.bias
        else:
            output = output
        
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
