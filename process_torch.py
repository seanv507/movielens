#%%
# https://github.com/Datadote/matrix-factorization-pytorch/blob/master/03_factorization_machines.ipynb
# Notebook influenced by this factorization machine post:
# yonigottesman.github.io/recsys/pytorch/elasticsearch/2020/02/18/fm-torch-to-recsys.html
# Changes include 1) LabelEncoder 2) refactored code 3) dataloaders + optimizer
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#%%
import wandb
#%%
# Data:     https://files.grouplens.org/datasets/movielens/ml-1m.zip
# Metadata: https://files.grouplens.org/datasets/movielens/ml-1m-README.txt
DATA_DIR = '../data/movielens/ml-1m/'
df_movies = pd.read_csv(DATA_DIR+'movies.dat', sep='::',
                        names=['movieId', 'title','genres'],
                        encoding='latin-1',
                        engine='python')
user_cols = ['userId', 'gender' ,'age', 'occupation', 'zipcode']
df_users = pd.read_csv(DATA_DIR+'users.dat', sep='::',
                       header=None,
                       names=user_cols,
                       engine='python')
df = pd.read_csv(DATA_DIR+'ratings.dat', sep='::',
                 names=['userId','movieId','rating','time'],
                 engine='python')
# Left merge removes movies with no rating. # of unique movies: 3883 -> 3706
df = df.merge(df_movies, on='movieId', how='left')
df = df.merge(df_users, on='userId', how='left')
df = df.sort_values(['userId', 'time'], ascending=[True, True]).reset_index(drop=True)
print(df.shape)
df.head(3)
#%%
plt.figure(figsize=(4, 1.5))
bins = np.arange(1, 6, 0.5) - 0.25
plt.hist(df.rating.values, bins=bins)
plt.xticks(np.arange(1, 5.01, 1))
plt.title('Num Ratings vs Rating')
plt.xlabel('Rating')
plt.ylabel('Num Ratings')
plt.grid()


# %%
# Convert users and movies into categorical - use LabelEncoder
# Column categorical range remapped between [0, len(df.column.unique())-1]
# Remapping is important to reduce memory size of nn.Embeddings
d = defaultdict(LabelEncoder)
cols_cat = ['userId', 'movieId', 'gender', 'age', 'occupation']
for c in cols_cat:
    d[c].fit(df[c].unique())
    df[c+'_index'] = d[c].transform(df[c])
    print(f'# unique {c}: {len(d[c].classes_)}')

min_num_ratings = df.groupby(['userId'])['userId'].transform(len).min()
print(f'Min # of ratings per user: {min_num_ratings}')
print(f'Min/Max rating: {df.rating.min()} / {df.rating.max()}')
print(f'df.shape: {df.shape}')
df.head(3)

#%%
# To use 1 embedding matrix, need to calculate & add offsets to each feature column
# Orig. paper uses 1-hot encoding, here we use ordinal encoding
# Ordinal encoding reduces memory size. Important for train speed
feature_cols = ['userId_index', 'movieId_index', 'gender_index', 'age_index',
                'occupation_index']
#
# Get offsets
feature_sizes = {}
for feat in feature_cols:
    feature_sizes[feat] = len(df[feat].unique())
feature_offsets = {}
NEXT_OFFSET = 0
for k,v in feature_sizes.items():
    feature_offsets[k] = NEXT_OFFSET
    NEXT_OFFSET += v

# Add offsets to each feature column
for col in feature_cols:
    df[col] = df[col].apply(lambda x: x + feature_offsets[col])
print('Offset - feature')
for k, os in feature_offsets.items():
    print(f'{os:<6} - {k}')
df.head(3)


# %%
# Make train and val dataset. Use last 5 rated movies per user
# 6040 unique users. Each user has minimum 20 rated movies
THRES = 5
cols = ['rating', *feature_cols]
df_train = df[cols].groupby('userId_index').head(-THRES).reset_index(drop=True)
df_val = df[cols].groupby('userId_index').tail(THRES).reset_index(drop=True)
print(f'df_train shape: {df_train.shape}')
print(f'df_val shape: {df_val.shape}')
df_train.head(3)
#%%
class MovieDataset(Dataset):
    """ Movie DS uses x_feats and y_feat """
    def __init__(self, df, x_feats, y_feat):
        super().__init__()
        self.df = df
        self.x_feats = df[x_feats].values
        self.y_rating = df[y_feat].values
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return self.x_feats[idx], self.y_rating[idx]

BS = 1024
ds_train = MovieDataset(df_train, feature_cols, 'rating')
ds_val = MovieDataset(df_val, feature_cols, 'rating')
dl_train = DataLoader(ds_train, BS, shuffle=True, num_workers=2)
dl_val = DataLoader(ds_val, BS, shuffle=True, num_workers=2)

xb, yb = next(iter(dl_train))
print(xb.shape, yb.shape)
print(xb)
print(yb)
#%%

class FM(nn.Module):
    """ Factorization Machine + user/item bias, weight init., sigmoid_range 
        Paper - https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
    """
    def __init__(self, num_feats, emb_dim, bias, sigmoid, init_std):
        super().__init__()
        self.x_emb = nn.Embedding(num_feats, emb_dim)
        # embedding has indices as inputs rather than 1 hot coding x input  is implicitly 1
        # output is vector for each feature
        self.bias = bias
        self.sigmoid = sigmoid
        self.init_std = init_std
        self.x_emb.weight.data.normal_(0,init_std)
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
        if self.sigmoid:
            return self.sigmoid_range(fm_out, low=0.5) # -> [bs]
        return fm_out

    def sigmoid_range(self, x, low=0, high=5.5):
        """ Sigmoid function with range (low, high) """
        return torch.sigmoid(x) * (high-low) + low

CFG = {
    'lr': 0.001,
    'num_epochs': 100,
    'weight_decay': 0.1,
    'emb_dim': 100,
    'sigmoid': False,
    'bias': True,
    'init_std': 0.1,
}

n_feats = int(pd.concat([df_train, df_val]).max().max())
n_feats = n_feats + 1 # "+ 1" to account for 0 - indexing
mdl = FM(n_feats, 
         emb_dim=CFG['emb_dim'],
         init_std=CFG['init_std'], 
         bias=CFG['bias'], 
         sigmoid=CFG['sigmoid'])
mdl.to(device)
opt = optim.AdamW(mdl.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
loss_fn = nn.MSELoss()
print(f'Model weights: {list(dict(mdl.named_parameters()).keys())}')
wandb.init(project="matrix_fact_0", config = CFG)

#%%
epoch_train_losses, epoch_val_losses = [], []

for i in range(CFG['num_epochs']):
    train_losses, val_losses = [], []
    mdl.train()
    for xb,yb in dl_train:
        xb, yb = xb.to(device), yb.to(device, dtype=torch.float)
        preds = mdl(xb)
        loss = loss_fn(preds, yb)
        train_losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    mdl.eval()
    for xb,yb in dl_val:
        xb, yb = xb.to(device), yb.to(device, dtype=torch.float)
        preds = mdl(xb)
        loss = loss_fn(preds, yb)
        val_losses.append(loss.item())
    # Start logging
    epoch_train_loss = np.mean(train_losses)
    epoch_val_loss = np.mean(val_losses)
    epoch_train_losses.append(epoch_train_loss)
    epoch_val_losses.append(epoch_val_loss)
    s = (f'Epoch: {i}, Train Loss: {epoch_train_loss:0.2f}, '
         f'Val Loss: {epoch_val_loss:0.2f}'
        )
    print(s)


# %%
