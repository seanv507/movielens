#%%
from time import perf_counter
import os
from sklearn.model_selection import train_test_split

import polars as pl
import pandas as pd
import plotnine as p9
from mizani.formatters import percent_format
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error, r2_score
from fastFM import als
import wandb
from scipy.optimize import least_squares
#%%
    


#%%
ratings = pl.read_parquet("../data/movielens/ml-10M100K/ratings.parquet")
# ratings = pl.read_csv(
#     "../data/movielens/ml-10M100K/ratings.dat",
#     separator=":", #polars doesn't support double char sep
#     has_header=False,
#     columns=[0,2,4,6],
#     new_columns=["user","item","rating","timestamp"]
# ).with_columns((pl.col("timestamp")*1e6).cast(pl.Datetime))
# ratings.write_parquet("../data/movielens/ml-10M100K/ratings.parquet")
#%%
ratings = pl.read_csv("../data/movielens/ml-100k/u1.base",
    separator="\t", #polars doesn't support double char sep
    has_header=False,
    new_columns=["user","item","rating","timestamp"]
).with_columns((pl.col("timestamp")*1e6).cast(pl.Datetime))
# u1 test is first 20000 data points, u2 is 20000-30000 etc
#%% [markdown]
# 20% of users have 55% of ratings
#
# 20% of movies have 65% of ratings
user_ratings = (
    ratings
    .to_series(0)
    .value_counts(sort=True, parallel=True)
    .with_columns(
        rank=pl.int_range(1,pl.count()+1),
        cum_sum=pl.col("count").cum_sum()
    )
    .with_columns(
        rank_prop = pl.col("rank")/pl.col("rank").tail(1),
        cum_prop = pl.col("cum_sum")/pl.col("cum_sum").tail(1)
    )
)
movie_ratings = (
    ratings
    .to_series(1)
    .value_counts(sort=True, parallel=True)
    .with_columns(
        rank=pl.int_range(1,pl.count()+1),
        cum_sum=pl.col("count").cum_sum()
    )
    .with_columns(
        rank_prop = pl.col("rank")/pl.col("rank").tail(1),
        cum_prop = pl.col("cum_sum")/pl.col("cum_sum").tail(1),
    )
)

n_user = len(user_ratings)
n_item = len(movie_ratings)
print (f"users: {n_user}, items: {n_item}  sparsity {user_ratings['count'].sum()/(n_user * n_item):.2%}")
plot_df  = (
    pl.concat((
        movie_ratings.select(["rank_prop","cum_prop"]).with_columns(type=pl.lit("movie")),
        user_ratings.select(["rank_prop","cum_prop"]).with_columns(type=pl.lit("user"))
    ))
)
(
    p9.ggplot(
        plot_df,
        p9.aes(x="rank_prop",y="cum_prop", colour="type"))
    + p9.geom_line() + 
    p9.scale_x_continuous(breaks=np.linspace(0,1,11), labels=percent_format()) 
    + p9.scale_y_continuous(breaks=np.linspace(0,1,11), labels=percent_format())
)
#%%
assert ratings["item"].max() < 100_000
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

data, y = process_fastfm.make_dataset()
#%%
X_train, X_test, y_train, y_test = train_test_split(data, y,test_size=0.1)

# %%
wandbkey = os.environ["WANDB_API_KEY"]
wandb.login(key=wandbkey)
#%%

#%%
def do_run(config):
    wandb_kwargs = {"project": "fastfm_movielens_03"}
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
    n_iter = model_config["n_iter"]
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

#%%

data_config = {
        "dataset": "10M",
        "split": "10%",
}
n_iter=10
model_config = {
    "optim": "als", "n_iter": n_iter, "init_stdev": 0.1, "rank": 10, "l2_reg_w": 10, "l2_reg_V": 10,
}
for rank in range(16,33,16):
    model_config["rank"] = rank
    for l2_w in [2,4,6,8, 10]:
        for l2_V in [2, 4,6,8, 10]:
            model_config["l2_reg_w"] = l2_w
            model_config["l2_reg_V"] = l2_V
            config = {"data": data_config, "model": model_config}
            print(rank, l2_w)
            do_run(config)
wandb.finish()
# %%
from fastFM.datasets import make_user_item_regression

X, y, coef = make_user_item_regression(n_user=100, n_item=100, rank=2,
                                           mean_w=0,mean_V=0 )
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

from fastFM.als import FMRegression
fm = FMRegression(rank=2,)

fm.fit(sp.csc_matrix(X_train), y_train)
#%%
y_pred = fm.predict(sp.csc_matrix(X_test))
y_pred_train = fm.predict(sp.csc_matrix(X_train))
print('rmse train', mean_squared_error(y_pred_train, y_train))
print('rmse', mean_squared_error(y_pred, y_test))
print('r2_score', r2_score(y_pred, y_test))
# np.random.shuffle(y_pred)
# print('----  shuffled pred ---------')
# print('rmse', mean_squared_error(y_pred, y_test))
# print('r2_score', r2_score(y_pred, y_test))
df_plot = pd.DataFrame({"pred":y_pred, "actual": y_test})
p9.ggplot(df_plot.sample(1000),p9.aes(x="pred", y="actual"))+p9.geom_point(alpha=0.2)

# %%[markdown]
# Finding Interaction Matrix
rng = np.random.default_rng()
mu_M = 0
std_M = 1
M_u = 20
M_i = 20
M_ui = M_u + M_i
n_factors = 3 
M = rng.normal(mu_M, std_M,(M_ui, M_ui))
M = .5 * (M + M.T)

#want a V: 

def residual_function(x):
    V = x.reshape(M_ui, n_factors)
    M_hat = V @ V.T
    
    residuals = (M - M_hat)[:M_u, M_u:M_u+M_i ].flatten()
    return residuals


def residual_all_function(x):
    V = x.reshape(M_ui, n_factors)
    M_hat = V @ V.T
    
    residuals = (M - M_hat).flatten()
    return residuals
#%%

V_0 = rng.normal(mu_M, std_M,size=int(M_ui * n_factors))
solution = least_squares(residual_function, V_0, verbose=2, max_nfev=100)
#%%
solution1 = least_squares(residual_all_function, V_0, verbose=2, max_nfev=100)
#%%
V = solution1["x"].reshape(M_ui, n_factors)

M_hat = V @ V.T
residuals = (M - M_hat)[:M_u, M_u:M_u+M_i ]
norm = (M[:M_u, M_u:M_u+M_i ]**2).mean() # we use mean to compare between diff data size
norm_hat = (M_hat[:M_u, M_u:M_u+M_i ]**2).mean() # we use mean to compare between diff data size
ev = np.linalg.eigvals(M_hat)
rmse = np.sqrt((residuals**2).mean()) 
print(f"rmse {rmse}, norm {norm:0.3f} norm_hat {norm_hat:0.3f} min ev m_hat {ev.min()}")

# %%
