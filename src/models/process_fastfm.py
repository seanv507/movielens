"""
An example based off the MovieLens 20M dataset.

This code will automatically download a HDF5 version of this
dataset when first run. The original dataset can be found here:
https://grouplens.org/datasets/movielens/.

Since this dataset contains explicit 5-star ratings, the ratings are
filtered down to positive reviews (4+ stars) to construct an implicit
dataset
"""


import logging
import time
from time import perf_counter
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import polars as pl
from fastFM import als
import wandb


logging.basicConfig()
log = logging.getLogger("fastfm")
log.setLevel(level=logging.DEBUG)


def objective(trial):
    config = make_config(trial)
    value = do_run(config)
    return value

def make_config(trial):
    
    data_config = {
        "dataset": "10M",
        "split": "10%",
    }
   
    model_config = {}

    model_config["optim"] = "als"
    model_config["n_iter"] = trial.suggest_categorical("n_iter",[10])
    model_config["rank"] = trial.suggest_categorical("embedding dimension", [1,2,4,8,16,32,64,128,])
    model_config["init_stdev"] = 0.1, 
    model_config["l2_reg_w"]: trial.suggest_categorical("l2_reg_w", [2,4,8,16])
    model_config["l2_reg_V"]: trial.suggest_categorical("l2_reg_V", [2,4,8,16])
    cfg = {
        "data": data_config,
        "model": model_config,
    }
    return cfg

def do_run(config):
    test_size = 0.10
    wandb_kwargs = {"project": "fastfm_movielens_04"}
    
    wandb_kwargs["config"] = config
    wandb.init(**wandb_kwargs)

    model_config = config["model"].copy()
    model_config.pop("optim")
    n_iter = model_config["n_iter"]
    fm = als.FMRegression(**model_config)
    data_config =  config["data"]
    data, y = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(data, y,test_size=test_size)

    iter=0
    for i in range(5):
        start = perf_counter()
        if iter:
            fm.fit(X_train,y_train,n_more_iter=n_iter)
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
        wandb.log(result, step = iter)
    return np.sqrt(mse_test)


def make_dataset():
    ratings = pl.read_parquet("s3://paprikadata/movielens/ml-10M100K/ratings.parquet")
    max_item = ratings["item"].max()
    user_offset = int(10**(np.ceil(np.log(max_item)/np.log(10))))
    rating_sp = pl.concat((
        pl.DataFrame({
            "row": np.arange(len(ratings)),
            "col": ratings["user"]+user_offset,
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
    return data, y

