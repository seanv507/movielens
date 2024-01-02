#%%
from time import perf_counter
import os
from sklearn.model_selection import train_test_split

# import xlearn as xl
import polars as pl
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error, r2_score
import fastFM
from fastFM import als
import wandb
#%%
    


#%%
# ratings = pl.read_csv(
#     "../data/movielens/ml-10M100K/ratings.dat",
#     separator=":", #polars doesn't support double char sep
#     has_header=False,
#     columns=[0,2,4,6],
#     new_columns=["user","item","rating","timestamp"]
# ).with_columns((pl.col("timestamp")*1e6).cast(pl.Datetime))
# ratings.write_parquet("../data/movielens/ml-10M100K/ratings.parquet")
ratings = pl.read_parquet("../data/movielens/ml-10M100K/ratings.parquet")
#%%
ratings = pl.read_csv("../data/movielens/ml-100k/u1.base",
    separator="\t", #polars doesn't support double char sep
    has_header=False,
    new_columns=["user","item","rating","timestamp"]
).with_columns((pl.col("timestamp")*1e6).cast(pl.Datetime))
# u1 test is first 20000 data points, u2 is 20000-30000 etc
#%% 


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
data = (
    sp.csc_matrix((rating_sp[:,2],(rating_sp[:,0], rating_sp[:,1])))
)
y = ratings["rating"].to_numpy()
#%%

X_train, X_test, y_train, y_test = train_test_split(data, y)

# %%
wandbkey = os.environ["WANDB_API_KEY"]
wandb.login(key=wandbkey)
#%%

#%%
def do_run(config):
    wandb_kwargs = {"project": "fastfm_movielens_02"}
    # wandb_config={}
    # for section in config.values():
    #     for key, value in section.items():
    #         wandb_config[key] = value
    wandb_kwargs["config"] = config
    wandb.init(**wandb_kwargs)

    model_config = config["model"].copy()
    model_config.pop("optim")
    data_config =  config["data"]

    fm = als.FMRegression(**model_config)            
    iter=0
    for i in range(20):
        start = perf_counter()
        if iter:
            fm.fit(X_train,y_train,n_more_iter=model_config["n_iter"])
        else:
            fm.fit(X_train,y_train)
        timing = perf_counter() - start
        iter += n_iter
        y_pred_train = fm.predict(X_train)
        mse_train = mean_squared_error(y_pred_train,y_train)
        y_pred_test = fm.predict(X_test)
        mse_test = mean_squared_error(y_pred_test,y_test)        
        result = {"iter": iter,
                  "timing": timing,
             "w_l2": np.linalg.norm(fm.w_), 
             "V_l2": np.linalg.norm(fm.V_), 
             "train":{"rmse":np.sqrt(mse_train)},
             "test":{"rmse":np.sqrt(mse_test)},
        }
        wandb.log(result)
    return np.sqrt(mse_test)

#%%

data_config = {
        "dataset": "100k",
        "split": "u1base",
}
n_iter=10
model_config = {
    "optim": "als", "n_iter": n_iter, "init_stdev": 0.1, "rank": 10, "l2_reg_w": 10, "l2_reg_V": 10,
}
for rank in range(4,8):
    model_config["rank"] = rank
    for l2_w in [2,4,8,16]:
        for l2_V in [2,4,8,16]:
            model_config["l2_reg_w"] = l2_w
            model_config["l2_reg_V"] = l2_V
            config = {"data": data_config, "model": model_config}
            print(rank, l2_w)
            do_run(config)
wandb.finish()
# %%
