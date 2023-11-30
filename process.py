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
../../fastFM/fastFM-core/bin/fastfm --rng-seed=123  --task=regression --init-var=0.1 --n-iter=50 --step-size=0.01 --solver=sgd --rank=32 --l2-reg-V=0.2 -l2-reg-w=0.2 --test-predict=test.txt  --verbose


     --rng-seed=NUM         Seed for random number generator (default current
                             time(NULL))
      --train-pairs=FILE     Ranking only! Required training pairs for bpr
                             training.
  -t, --task=S               The tasks: 'classification', 'regression' or
                             'ranking' are supported (default 'regression').
                             Ranking uses the Bayesian Pairwise Ranking (BPR)
                             loss and needs an additional file (see
                             '--train-pairs')

 Solver:
  -i, --init-var=NUM         N(0, var) is used to initialize the coefficients
                             of matrix V (default 0.1)
  -n, --n-iter=NUM           Number of iterations (default 50)
      --step-size=NUM        Step-size for 'sgd' updates (default 0.01)
  -s, --solver=S             The solvers: 'als', 'mcmc' and 'sgd' are available
                             for 'regression' and 'classification (default
                             'mcmc'). Ranking is only supported by 'sgd'.

 Model Complexity and Regularization:
  -k, --rank=NUM             Rank of the factorization, Matrix V (default 8).
      --l2-reg-V=NUM         l2 regularization for the latent representation
                             (V) of the pairwise coefficients
      --l2-reg-w=NUM         l2 regularization for the linear coefficients (w)
  -r, --l2-reg=NUM           l2 regularization, set equal penalty for all
                             coefficients (default 1)

 I/O options:
  -7, --test-predict=FILE    Save prediction from TEST_FILE to FILE.

 Informational Options:
  -q, --quiet, --silent      Don't produce any output
  -v, --verbose              Produce verbose output

  -?, --help                 Give this help list
      --usage                Give a short usage message
  -V, --version              Print program version
