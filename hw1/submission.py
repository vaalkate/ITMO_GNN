# submission.py
import torch
import torch.nn as nn
from torch_geometric.utils import add_self_loops, degree


def _scatter_add(src, index, dim_size):
    out = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


def _scatter_count(index, dim_size, device, dtype):
    ones = torch.ones(index.numel(), 1, dtype=dtype, device=device)
    out = torch.zeros(dim_size, 1, dtype=dtype, device=device)
    out.index_add_(0, index, ones)
    return out


class MyGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        x = self.lin(x)
        return _scatter_add(x[row] * norm.view(-1, 1), col, x.size(0))


class MySAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        nn.init.xavier_uniform_(self.lin_self.weight)
        if self.lin_neigh.bias is not None:
            nn.init.zeros_(self.lin_neigh.bias)
            nn.init.zeros_(self.lin_self.bias)

    def forward(self, x, edge_index):
        row, col = edge_index
        out_neigh = _scatter_add(x[row], col, x.size(0))
        deg = _scatter_count(col, x.size(0), x.device, x.dtype).clamp(min=1)
        out_neigh = out_neigh / deg
        return self.lin_neigh(out_neigh) + self.lin_self(x)