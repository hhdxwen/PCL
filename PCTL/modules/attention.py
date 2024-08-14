import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_in % num_heads == 0 and dim_in % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_in
        self.dim_v = dim_in
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_in, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_in, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_in, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_in // num_heads)

    def forward(self, x, rela):
        # x: tensor of shape (n, dim_in)
        n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(n, nh, dk).transpose(0, 1)  # (nh, n, dk)
        k = self.linear_k(x).reshape(n, nh, dk).transpose(0, 1)  # (nh, n, dk)
        v = self.linear_v(x).reshape(n, nh, dv).transpose(0, 1)  # (nh, n, dv)

        dist = torch.matmul(q, k.transpose(1, 2)) * self._norm_fact  # nh, n, n
        dist = torch.softmax(dist,dim=-1)  #nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(0, 1).reshape( n, self.dim_v)  # n, dim_v
        return att