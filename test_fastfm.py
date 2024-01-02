#!/usr/bin/env python
#%%
from sklearn.model_selection import train_test_split

# import xlearn as xl
import polars as pl
import pandas as pd
import numpy as np
import scipy.sparse as sp
import fastFM
#%%
ratings = pl.read_parquet("../data/movielens/ml-10M100K/ratings.parquet")
#%%
rating_sp = pl.concat((
    pl.DataFrame({
        "row": np.arange(len(ratings)),
        "col": ratings["user"]+100_000,
        "value":1,}),
    pl.DataFrame({
        "row": np.arange(len(ratings)),
        "col": ratings["item"],
        "value":1,})
)).to_numpy()
#%%
data = (
    sp.csc_matrix((rating_sp[:,2],(rating_sp[:,0], rating_sp[:,1])))
)
y = ratings["rating"].to_numpy()
#%%

# This sets up a small test dataset.

X_train, X_test, y_train, y_test = train_test_split(data, y)

# %%
from fastFM import als
fm = als.FMRegression(n_iter=10, init_stdev=0.1,rank=2, l2_reg_w=0.1, l2_reg_V=0.5)
# %%
fm.fit(X_train,y_train)
# %%
