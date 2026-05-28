import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from typing import Optional
from lib_models.HNN.utils import zeros,glorot

class HypergraphConv(MessagePassing):

    def __init__(self, in_channels, out_channels, symdegnorm=False, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 attention_mode='edge',
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HypergraphConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.symdegnorm = symdegnorm
        self.attention_mode = attention_mode
        if self.use_attention and self.attention_mode not in ['node', 'edge']:
            raise ValueError("attention_mode must be either 'node' or 'edge'")

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = nn.Parameter(
                torch.Tensor(in_channels, heads * out_channels))
            self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix :math:`\mathbf{X}`
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Sparse hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges,device=x.device)

        x = torch.matmul(x, self.weight)

        alpha = None
        if self.use_attention:
            x = x.view(-1, self.heads, self.out_channels)
            edge_card = scatter_add(
                x.new_ones(hyperedge_index.size(1), device=x.device),
                hyperedge_index[1],
                dim=0,
                dim_size=num_edges,
            ).clamp(min=1.0)
            hyperedge_attr = scatter_add(
                x[hyperedge_index[0]],
                hyperedge_index[1],
                dim=0,
                dim_size=num_edges,
            ) / edge_card.view(-1, 1, 1)
            x_i, x_j = x[hyperedge_index[0]], hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            if self.attention_mode == 'node':
                alpha = softmax(alpha, hyperedge_index[1], num_nodes=num_edges)
            else:
                alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if not self.symdegnorm:
            D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                            hyperedge_index[0], dim=0, dim_size=num_nodes)
            D = 1.0 / D
            D[D == float("inf")] = 0

            B = scatter_add(x.new_ones(hyperedge_index.size(1),device=x.device),
                            hyperedge_index[1], dim=0, dim_size=num_edges)  
            B = 1.0 / B
            B[B == float("inf")] = 0
            
            self.flow = 'source_to_target'
            edge_embed = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                                 size=(num_nodes, num_edges))
            self.flow = 'target_to_source'
            out = self.propagate(hyperedge_index, x=edge_embed, norm=D, alpha=alpha,size=(num_nodes, num_edges))
            
        else:  # this correspond to HGNN
            D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                            hyperedge_index[0], dim=0, dim_size=num_nodes)
            D = 1.0 / D**(0.5)
            D[D == float("inf")] = 0

            B = scatter_add(x.new_ones(hyperedge_index.size(1),device=x.device),
                            hyperedge_index[1], dim=0, dim_size=num_edges)
            B = 1.0 / B
            B[B == float("inf")] = 0

            x = D.unsqueeze(-1)*x
            self.flow = 'source_to_target'
            edge_embed = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                                 size=(num_nodes, num_edges))

            self.flow = 'target_to_source'
            out = self.propagate(hyperedge_index,x=edge_embed, norm=D, alpha=alpha,size=(num_nodes, num_edges))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
            edge_embed = edge_embed.view(-1,self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            edge_embed = edge_embed.mean(dim=-1)

        if self.bias is not None:
            out = out + self.bias

        return out,edge_embed 

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels
        
        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class HCHA(nn.Module):

    def __init__(self, num_features, num_targets, args):
        super(HCHA, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout  # Note that default is 0.6
        self.symdegnorm = args.HCHA_symdegnorm
        self.hidden_dim = args.MLP_hidden
        self.use_attention = getattr(args, "HCHA_use_attention", False)
        self.heads = getattr(args, "HCHA_heads", 8)
        self.output_heads = getattr(args, "HCHA_output_heads", 1)
        self.attention_mode = getattr(args, "HCHA_attention_mode", "edge")
        self.negative_slope = getattr(args, "HCHA_negative_slope", 0.2)

#       Note that add dropout to attention is default in the original paper
        self.convs = nn.ModuleList()

        def layer(in_channels, out_channels, heads):
            if self.use_attention:
                if out_channels % heads != 0:
                    raise ValueError(
                        f"HCHA attention output dimension {out_channels} must be divisible by heads={heads}"
                    )
                return HypergraphConv(
                    in_channels,
                    out_channels // heads,
                    self.symdegnorm,
                    use_attention=True,
                    heads=heads,
                    concat=True,
                    negative_slope=self.negative_slope,
                    dropout=self.dropout,
                    attention_mode=self.attention_mode,
                )
            return HypergraphConv(in_channels, out_channels, self.symdegnorm)

        if self.num_layers == 1:
            self.convs.append(layer(num_features, num_targets, self.output_heads))
        else:
            self.convs.append(layer(num_features, self.hidden_dim, self.heads))
            for _ in range(self.num_layers-2):
                self.convs.append(layer(self.hidden_dim, self.hidden_dim, self.heads))
            # Output heads is set to 1 as default
            self.convs.append(layer(self.hidden_dim, num_targets, self.output_heads))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):

        # regular node classification

        x = data.x
        edge_index = data.hyperedge_index

        for i, conv in enumerate(self.convs[:-1]):
            x , e = conv(x,edge_index) 
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x,e = self.convs[-1](x, edge_index)

        return x,e

    @torch.no_grad()
    def predict(self,data):
        self.eval()
        return self.forward(data)
