import torch
import torch.nn as nn
from collections import defaultdict
from lib_models.HNN import MLP
from torch_geometric.nn import global_max_pool,global_mean_pool
from torch import Tensor


def safe_scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int = None):

    if dim < 0:
        dim = src.dim() + dim

    if dim_size is None:
        dim_size = index.max().item() + 1

    for _ in range(src.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand_as(src)

    out_shape = list(src.shape)
    out_shape[dim] = dim_size

    out = torch.full(out_shape, float('-inf'), dtype=src.dtype, device=src.device)

    out = out.scatter_reduce(dim=dim, index=index, src=src, reduce='amax', include_self=True)

    return out

# -------------------------------- Hypergraph Classification Aggregator ------------------------------

class HyperGPredictor(nn.Module):
    def __init__(self, encoder, num_targets, args):
        super(HyperGPredictor, self).__init__()
        
        self.args=args
        self.encoder = encoder
        
        self.pooling = args.pooling 
        self.g_embed_hidden=args.g_embed_hidden
        self.g_embed_layer=args.g_embed_layer
        self.g_embed_dropout=args.g_embed_dropout
        self.g_embed_norm = args.g_embed_norm
        
        self.classifier = MLP(in_channels=args.embedding_hidden,
                hidden_channels=self.g_embed_hidden,
                out_channels=num_targets,
                num_layers=self.g_embed_layer,
                dropout=self.g_embed_dropout,
                Normalization=self.g_embed_norm,
                InputNorm=False)
    
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()
    
    def forward(self,data):
        
        n_emb,_ = self.encoder(data)

        if self.pooling == 'mean':
            g_emb = global_mean_pool(n_emb,data.batch)
        elif self.pooling == 'max':
            #g_emb = global_max_pool(n_emb,data.batch)
            dim = -1 if isinstance(n_emb, Tensor) and n_emb.dim() == 1 else -2
            g_emb = safe_scatter_max(n_emb, data.batch, dim=dim)
        
        g_emb = self.classifier(g_emb)
        
        return g_emb

# -------------------------------- Hyperedge Prediction Aggregator ------------------------------

class EdgePredictor(nn.Module):
    def __init__(self, encoder, aggregator,args):
        super(EdgePredictor, self).__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.args=args
        self.edge_aggr = args.edge_aggr 
    
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.aggregator.reset_parameters()
        
    def encoding(self,data):
        n_embed = self.encoder(data)
        return n_embed

    def aggregate(self, nfeat, hedges, mode='Train'):
        if self.edge_aggr == 'group':
            size_groups = defaultdict(list)
            for he in hedges:
                size_groups[len(he)].append(he)

            preds = []
            for size, group in size_groups.items():
                he_feats = torch.stack([nfeat[he] for he in group])  # (num_hedges, size, feature_dim)
                pred = self.aggregator(he_feats,self.edge_aggr)
                preds.append(pred)
            
            if mode == 'Train':
                return torch.cat([seq.squeeze(-1) for seq in preds],dim=0)
            elif mode == 'Eval':
                final_preds=torch.cat([seq for seq in preds],dim=0)
                return [pred.detach() for pred in final_preds]
            else:
                raise NotImplementedError
        else:
            preds = []
            if mode == 'Train':
                for he in hedges:
                    feat = nfeat[he]
                    pred = self.aggregator(feat,self.edge_aggr)
                    preds.append(pred)
                return torch.stack(preds).squeeze()

            elif mode == 'Eval':
                for he in hedges:
                    feat = nfeat[he]
                    pred = self.aggregator(feat,self.edge_aggr)
                    preds.append(pred.detach())
                return preds
            else:
                raise NotImplementedError

class MeanAggregator(nn.Module):
    def __init__(self, args):
        super(MeanAggregator, self).__init__()

        self.e_embed_hidden=args.e_embed_hidden
        self.e_embed_layer=args.e_embed_layer
        self.e_embed_dropout=args.e_embed_dropout
        self.e_embed_norm = args.e_embed_norm
        
        self.classifier = MLP(in_channels=args.embedding_hidden,
                hidden_channels=self.e_embed_hidden,
                out_channels=1,
                num_layers=self.e_embed_layer,
                dropout=self.e_embed_dropout,
                Normalization=self.e_embed_norm,
                InputNorm=False)

    def reset_parameters(self):

        self.classifier.reset_parameters()
    
    def aggregate(self, embeddings, method='group'):
        dim=1 if method == 'group'else 0
        embedding = embeddings.mean(dim=dim).squeeze()
        return embedding
    
    def classify(self, embedding) :
#         pdb.set_trace()
        embedding = torch.linalg.norm(embedding.unsqueeze(0),dim=0)      
        return self.classifier(embedding)
    
    def forward(self, embeddings, method='group'):
        embedding = self.aggregate(embeddings,method)
        pred = self.classify(embedding)
        return pred

class MaxminAggregator(nn.Module):
    def __init__(self, args):
        super(MaxminAggregator, self).__init__()

        self.e_embed_hidden=args.e_embed_hidden
        self.e_embed_layer=args.e_embed_layer
        self.e_embed_dropout=args.e_embed_dropout
        self.e_embed_norm = args.e_embed_norm
        
        self.classifier = MLP(in_channels=args.embedding_hidden,
                hidden_channels=self.e_embed_hidden,
                out_channels=1,
                num_layers=self.e_embed_layer,
                dropout=self.e_embed_dropout,
                Normalization=self.e_embed_norm,
                InputNorm=False)

    def reset_parameters(self):

        self.classifier.reset_parameters()
    
    def aggregate(self, embeddings, method='group'):
        dim=1 if method == 'group'else 0
        max_val, _ = torch.max(embeddings, dim=dim) 
        min_val, _ = torch.min(embeddings, dim=dim)
        return max_val - min_val
    
    def classify(self, embedding):
        return self.classifier(embedding)
    
    def forward(self, embeddings,method):
        embedding = self.aggregate(embeddings,method)
        pred = self.classify(embedding)
        return pred

class MaxAggregator(nn.Module):
    def __init__(self, args):
        super(MaxAggregator, self).__init__()

        self.e_embed_hidden=args.e_embed_hidden
        self.e_embed_layer=args.e_embed_layer
        self.e_embed_dropout=args.e_embed_dropout
        self.e_embed_norm = args.e_embed_norm
        
        self.classifier = MLP(in_channels=args.embedding_hidden,
                hidden_channels=self.e_embed_hidden,
                out_channels=1,
                num_layers=self.e_embed_layer,
                dropout=self.e_embed_dropout,
                Normalization=self.e_embed_norm,
                InputNorm=False)

    def reset_parameters(self):

        self.classifier.reset_parameters()
    
    def aggregate(self, embeddings, method='group'):
        dim=1 if method == 'group'else 0
        max_val, _ = torch.max(embeddings, dim=dim) 
        return max_val
    
    def classify(self, embedding):
        return self.classifier(embedding)
    
    def forward(self, embeddings,method):
        embedding = self.aggregate(embeddings,method)
        pred = self.classify(embedding)
        return pred
