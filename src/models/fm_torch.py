
import torch
import torch.nn.functional as F
from torch import nn

class FM(nn.Module):
    """ Factorization Machine + user/item bias, weight init., sigmoid_range 
        Paper - https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
    """
    def __init__(self, num_feats, emb_dim, bias, init_std, seed):
        super().__init__()
        self.x_emb = nn.Embedding(num_feats, emb_dim)
        # embedding has indices as inputs rather than 1 hot coding x input  is implicitly 1
        # output is vector for each feature
        self.bias = bias
        self.init_std = init_std
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None
        self.x_emb.weight.data.normal_(0,init_std, generator=generator)

        if bias:
            self.x_bias = nn.Parameter(torch.zeros(num_feats))
            self.offset = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        # Derived time complexity - O(nk)
        x_emb = self.x_emb(X) # [bs, num_feats] -> [bs, num_feats, emb_dim] $\v_{i,f}x_i$
        pow_of_sum = x_emb.sum(dim=1).pow(2) # -> [bs, num_feats]
        sum_of_pow = x_emb.pow(2).sum(dim=1) # -> [bs, num_feats]
        fm_out = (pow_of_sum - sum_of_pow).sum(1)*0.5  # -> [bs]
        if self.bias:
            x_biases = self.x_bias[X].sum(1) # -> [bs]
            fm_out +=  x_biases + self.offset # -> [bs]
        return fm_out

