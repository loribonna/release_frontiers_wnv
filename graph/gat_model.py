import torch
import torch.nn as nn
from graph.layers_gat import GraphAttentionLayer
from graph.layers_gat import SinkhornKnopp


class MAGAT(nn.Module):
    """Dense version of GAT."""

    def __init__(self, nin, nhid, nout, alpha, nheads=1):
        super(MAGAT, self).__init__()

        self.attn1 = nn.ModuleList([GraphAttentionLayer(nin, nhid, alpha=alpha,
                                                        concat=True) for _ in range(nheads)])
        self.attn2 = nn.ModuleList([GraphAttentionLayer(nhid * nheads, nhid, alpha=alpha,
                                                        concat=True) for _ in range(nheads)])
        self.graph_norm = SinkhornKnopp()
        self.act = torch.nn.ELU()
        self.dropout = nn.Dropout(p=0.2)

    def forward_to_graph_attn_layer(self, x: torch.Tensor, adj: torch.Tensor,
                                    m: torch.nn.Module):
        x_res, adj_res = [], []
        for i, m in enumerate(m):
            x_i, adj_i = m(x, adj[i])
            x_res.append(x_i)
            adj_res.append(adj_i)
        x_res = torch.cat(x_res, 1)
        adj_res = torch.stack(adj_res)
        return x_res, adj_res

    def forward(self, x0, adj0):
        # DS normalization
        adj0 = torch.stack([self.graph_norm(G) for G in adj0])

        # First Gat Layer
        x1, _ = self.forward_to_graph_attn_layer(x0, adj0, self.attn1)

        x0 = x0.repeat(1, adj0.shape[0])
        x1 = x1 + x0
        return self.act(x1)
