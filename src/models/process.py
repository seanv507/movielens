# https://github.com/Datadote/matrix-factorization-pytorch/blob/master/03_factorization_machines.ipynb
# Notebook influenced by this factorization machine post:
# yonigottesman.github.io/recsys/pytorch/elasticsearch/2020/02/18/fm-torch-to-recsys.html
# Changes include 1) LabelEncoder 2) refactored code 3) dataloaders + optimizer
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import wandb
from .fm_torch import FM
from .fm_prod_torch import FM_prod

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



feature_cols = ['userId_index', 'movieId_index', 'gender_index', 'age_index',
                'occupation_index']


def get_movielens_1m():
    # Data:     https://files.grouplens.org/datasets/movielens/ml-1m.zip
    # Metadata: https://files.grouplens.org/datasets/movielens/ml-1m-README.txt
    
    LOCAL_DATA_DIR = './data/movielens/ml-1m/'
    df = None
    try:
        df = pd.read_parquet(LOCAL_DATA_DIR + "data.parquet")
    except FileNotFoundError:
        pass
    if df is not None:
        df = download_movielens_1m()
        df = convert_categorical(df)
        df = make_embedding_indices(df)
        p = Path(LOCAL_DATA_DIR)
        p.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p / "data.parquet")
    return df



def download_movielens_1m():
    S3_DATA_DIR = 's3://paprikadata/movielens/ml-1m/'
    df_movies = pd.read_csv(S3_DATA_DIR+'movies.dat', sep='::',
                            names=['movieId', 'title','genres'],
                            encoding='latin-1',
                            engine='python')
    user_cols = ['userId', 'gender' ,'age', 'occupation', 'zipcode']
    df_users = pd.read_csv(S3_DATA_DIR+'users.dat', sep='::',
                        header=None,
                        names=user_cols,
                        engine='python')
    df = pd.read_csv(S3_DATA_DIR+'ratings.dat', sep='::',
                    names=['userId','movieId','rating','time'],
                    engine='python')
    # Left merge removes movies with no rating. # of unique movies: 3883 -> 3706
    df = df.merge(df_movies, on='movieId', how='left')
    df = df.merge(df_users, on='userId', how='left')
    df = df.sort_values(['userId', 'time'], ascending=[True, True]).reset_index(drop=True)
    
    return df


def convert_categorical(df: pd.DataFrame):
    # Convert users and movies into categorical - use LabelEncoder
    # Column categorical range remapped between [0, len(df.column.unique())-1]
    # Remapping is important to reduce memory size of nn.Embeddings
    d = defaultdict(LabelEncoder)
    cols_cat = ['userId', 'movieId', 'gender', 'age', 'occupation']
    for c in cols_cat:
        d[c].fit(df[c].unique())
        df[c+'_index'] = d[c].transform(df[c])
        # print(f'# unique {c}: {len(d[c].classes_)}')
    return df


def make_embedding_indices(df):
    # To use 1 embedding matrix, need to calculate & add offsets to each feature column
    # Orig. paper uses 1-hot encoding, here we use ordinal encoding
    # Ordinal encoding reduces memory size. Important for train speed
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
    return df


def get_n_feats(df):
    n_feats = df[feature_cols].max().max() + 1 # "+ 1" to account for 0 - indexin
    return n_feats


def split_train_test(df):
    # Make train and val dataset. Use last 5 rated movies per user
    # 6040 unique users. Each user has minimum 20 rated movies
    THRES = 5
    cols = ['rating', *feature_cols]
    df_train = df[cols].groupby('userId_index').head(-THRES).reset_index(drop=True)
    df_val = df[cols].groupby('userId_index').tail(THRES).reset_index(drop=True)
    return df_train, df_val


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

def make_dataset(df):
    BS = 1024
    ds = MovieDataset(df, feature_cols, 'rating')
    dl = DataLoader(ds, BS, shuffle=True, num_workers=0)
    return dl


def create_model_setup(CFG, n_feats):
    models = {"FM": FM, "FM_prod": FM_prod}
 
    mdl = models[CFG['model']](
        n_feats, 
        emb_dim=CFG['emb_dim'],
        init_std=CFG['init_std'], 
        seed=CFG['seed'],
        bias=CFG['bias'], 
        )
    mdl.to(device)
    opt = optim.AdamW(mdl.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    loss_fn = nn.MSELoss()
    model_setup = {"model": mdl, "opt": opt, "loss_fn": loss_fn}
    return model_setup


def train_model(model, opt, loss_fn, dl_train, dl_val, ):
    train_losses, val_losses = [], []
    model.train()
    for xb,yb in dl_train:
        xb, yb = xb.to(device), yb.to(device, dtype=torch.float)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        train_losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    for xb,yb in dl_val:
        xb, yb = xb.to(device), yb.to(device, dtype=torch.float)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        val_losses.append(loss.item()) #?
    # Start logging
    epoch_train_loss = np.mean(train_losses)
    epoch_val_loss = np.mean(val_losses)
    
    wandb.log({"train_loss": epoch_train_loss, "val_loss": epoch_val_loss, })
    return model, epoch_train_loss, epoch_val_loss

