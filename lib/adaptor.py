import os
import sys
import glob
import h5py
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import datetime

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None, position_embedding=None):
    d_k = query.size(-1)
    
    # scores (b,h,n,n)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    if position_embedding is not None:
        position_embedding = position_embedding.unsqueeze(1) 
        scores = scores + position_embedding

    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, fn_attention=attention, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.fn_attention = fn_attention
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None, position_embedding=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.fn_attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PriorAdaptor(nn.Module):
    def __init__(self, emb_dims=512, n_heads=4):
        super(PriorAdaptor, self).__init__()
        self.enricher = MultiHeadedAttention(n_heads, emb_dims)

    def forward(self, *input):
        query = input[0]
        key = input[1]
        value = input[2]

        query = query.transpose(2, 1).contiguous()
        key = key.transpose(2, 1).contiguous()
        value = value.transpose(2, 1).contiguous()

        x = self.enricher(query, key, value).transpose(2, 1).contiguous()

        return x

if __name__ == '__main__':

    model = PriorAdaptor(emb_dims=128, n_heads=4).cuda()
    key = torch.randn(2, 128, 1024).cuda()
    query = torch.randn(2, 128, 1024).cuda()
    value = torch.randn(2, 128, 1024).cuda()

    embedding = model(query, key, value)

    print(embedding.shape)