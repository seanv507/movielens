#%%
from sklearn.model_selection import train_test_split

# import xlearn as xl
import polars as pl
import pandas as pd
import numpy as np
import scipy.sparse as sp
import fastFM
#%%
# ratings = pl.read_csv(
#     "../data/movielens/ml-10M100K/ratings.dat",
#     separator=":", #polars doesn't support double char sep
#     has_header=False,
#     columns=[0,2,4,6],
#     new_columns=["user","item","rating","timestamp"]
# ).with_columns((pl.col("timestamp")*1e6).cast(pl.Datetime))
#%%
#%%
# ratings.write_parquet("../data/movielens/ml-10M100K/ratings.parquet")
ratings = pl.read_parquet("../data/movielens/ml-10M100K/ratings.parquet")
#%%
# new_data = ratings.with_columns(
#     ("0:" + pl.col("user").cast(pl.Utf8)).alias("user"),
#     ("1:" + pl.col("item").cast(pl.Utf8)).alias("item"),
# ).select(["rating", "user","item"])
#%%
# create data in ffm format

# split into 90% train
# use 10 fold crossvalidation

# #rating format = userid:movieid::rating::timestamp

# # libffm format - what about cross terms?
# label field1:feat1:val1 field2:feat2:val2 ....
#%% 
# train_test_seed = 123
# new_data_shuffled = new_data.sample(fraction=1,shuffle=True, with_replacement=False, seed = train_test_seed)
# test_fraction = 0.1
# train_length =int(np.ceil(len(new_data_shuffled)*(1-test_fraction)))
# train_data = new_data_shuffled[:train_length]
# test_data = new_data_shuffled[train_length:]

# train_data.write_csv("data/train.txt", separator=" ",include_header=False)
# test_data.write_csv("data/test.txt", separator=" ",include_header=False)

#%%
# regularisation {0.02, 0.03, 0.04, 0.05}
# learning rates {0.001, 0.003}
# y = new_data["rating"].to_pandas()
# X = new_data[["user","item"]].to_pandas()
# train = xl.DMatrix(X,y)
# %%
# fm_model = xl.create_fm()
# # %%
# fm_model.setTrain("data/train.txt")
# fm_model.setTXTModel("data/model.txt")
# fm_model.setTest("data/test.txt")  
# # %%
# param = {'task':'reg', 'fold': 3, 'k': 128, 'lr':0.05, 'epoch': 3, 'lambda':0.02, 'metric':'rmse'}
# fm_model.fit(param,"data/model.out")
# fm_model.predict("data/model.out", "data/output.txt")
# # %%
# fm_model.predict("data/model.out", "data/output.txt")
# # xlearn is giving mse of .562 -> rmse 0.75 which is much lower than reported rmse

# # %%
# pred = pl.read_csv("data/output.txt",has_header=False,new_columns=["pred"])
# y_test = pl.read_csv("data/test.txt",has_header=False, columns=[0], new_columns=["actual"], separator=" ")
# # %%
# mse = ((pred["pred"] - y_test["actual"])**2).mean()
# print(f"rmse={np.sqrt(mse)}")
# #%%

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
