import numpy as np
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(torch.nn.functional.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class Attention(nn.Module):
    def __init__(self, d_model, d_k, d_v, dropout=0.1, n_head=1):
        super(Attention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.W_k = nn.Linear(in_features=d_model, out_features=d_k * n_head, bias=False)
        self.W_q = nn.Linear(in_features=d_model, out_features=d_k * n_head, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_v * n_head, bias=False)

        self.dot_product_attention = ScaledDotProductAttention(
            temperature=np.sqrt(d_k),
            attn_dropout=self.dropout
          )

    def forward(self, q, k, v, mask=None, return_attention=False):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.W_q(q).view(sz_b, len_q, n_head, d_k)
        k = self.W_k(k).view(sz_b, len_k, n_head, d_k)
        v = self.W_v(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        z, attn = self.dot_product_attention(q, k, v, mask=mask)
        if return_attention:
          return z, attn
        return z
