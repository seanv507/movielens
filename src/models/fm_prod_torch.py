
import torch
import torch.nn.functional as F
from torch import nn

class FM_prod(nn.Module):
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
        
        x_emb = self.x_emb(X) # [bs, num_feats] -> [bs, num_feats, emb_dim] $\v_{i,f}x_i$
        num_feats = x_emb.size(1)
        fm_out = 0
        for i_feat in range(num_feats):
            x_emb_i = x_emb[:,i_feat,:]
            for j_feat in range(i_feat):
                x_emb_j = x_emb[:,j_feat,:]
                fm_out += (x_emb_i * x_emb_j).sum(1)
        
        if self.bias:
            x_biases = self.x_bias[X].sum(1) # -> [bs]
            fm_out +=  x_biases + self.offset # -> [bs]
        return fm_out
