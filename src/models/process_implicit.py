"""
An example based off the MovieLens 20M dataset.

This code will automatically download a HDF5 version of this
dataset when first run. The original dataset can be found here:
https://grouplens.org/datasets/movielens/.

Since this dataset contains explicit 5-star ratings, the ratings are
filtered down to positive reviews (4+ stars) to construct an implicit
dataset
"""

from __future__ import print_function

import argparse
import codecs
import logging
import time

import numpy as np
import tqdm

import wandb

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.datasets.movielens import get_movielens
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import (
    BM25Recommender,
    CosineRecommender,
    TFIDFRecommender,
    bm25_weight,
)
from implicit.gpu import HAS_CUDA
from implicit.evaluation import leave_k_out_split, ranking_metrics_at_k

logging.basicConfig()
log = logging.getLogger("implicit")
log.setLevel(level=logging.DEBUG)


def objective(trial):
    cfg = set_cfg(trial)
    variant = cfg.pop("variant")
    model_kwargs = cfg
    value = benchmark_movies(variant=variant, model_kwargs=model_kwargs)
    return value

# factors=100,
# regularization=0.001,
# alpha=1, # The weight to give to positive examples.
# dtype=np.float32,
# use_native=True,
# use_cg=True,
# use_gpu=HAS_CUDA,
# iterations=15,
# calculate_training_loss=False,
def set_cfg(trial):
    cfg = {}
    cfg["variant"] = trial.suggest_categorical("data variant",["1m"])
    cfg["model_name"] = trial.suggest_categorical("model_name",["als"])
    cfg["use_gpu"] = trial.suggest_categorical("use gpu",[True])
    cfg["factors"] = trial.suggest_categorical("embedding dimension", [1,2,4,8,16,32,64,128,])
    cfg["regularization"] = trial.suggest_categorical("regularization", [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,])
    cfg["iterations"] = trial.suggest_categorical("iterations", [16,32,64,128])
    cfg["use_BM25"] = trial.suggest_categorical("use BM 25", [True, False])
    wandb.config = cfg
    return cfg


#"IndexError('index 3953 is out of bounds for axis 1 with size 3953')"

# 2023-12-18 18:04:09.9860

# 10.0.243.177

# [W 2023-12-18 17:04:09,986] Trial 99 failed with parameters: 
# {'data variant': '1m', 'model_name': 'als', 'embedding dimension': 16, 
#  'regularization': 1e-06, 'iterations': 128, 'use BM 25': True} 
#because of the following error: IndexError('index 3953 is out of bounds for axis 1 with size 3953').

def benchmark_movies(
        min_rating=4.0, 
        variant="20m",
        model_kwargs={}):
    
    model_name = model_kwargs.pop("model_name")
    use_bm25 = model_kwargs.pop("use_BM25", False)
    model_proc = {
        "als": AlternatingLeastSquares,
        "bpr": BayesianPersonalizedRanking,
        "lmf": LogisticMatrixFactorization,
        "tfidf": TFIDFRecommender,
        "cosine": CosineRecommender,
        "bm25": BM25Recommender, #B=0.2, **model_kwargs)
    }
    if model_name in model_proc.keys():
        # generate a recommender model based off the input params
        model = model_proc[model_name](
            **model_kwargs,
        )
    else:
        raise NotImplementedError(f"model {model_name} isn't implemented for this example")

    
    log.info("has cuda %s" % HAS_CUDA)
    
    # read in the input data file
    start = time.time()
    titles, ratings = get_movielens(variant)

    # remove things < min_rating, and convert to implicit dataset
    # by considering ratings as a binary preference only
    ratings.data[ratings.data < min_rating] = 0
    ratings.eliminate_zeros()
    ratings.data = np.ones(len(ratings.data))
    log.info("read data file in %s", time.time() - start)
    
    if model_name == "als":
        ratings_train, ratings_test = leave_k_out_split(ratings, K=5, train_only_size=0.0)    
        ratings = ratings_train # to not break rest of code
        # lets weight these models by bm25weight.
        if use_bm25:
            log.debug("weighting matrix by bm25_weight")
            ratings = (bm25_weight(ratings, B=0.9) * 5)
        ratings = ratings.tocsr()

    user_ratings = ratings.T.tocsr()

    # train the model
    log.debug("training model %s", model_name)
    start = time.time()
    model.fit(user_ratings)
    log.debug("trained model '%s' in %s", model_name, time.time() - start)
    rankings = ranking_metrics_at_k(model, ratings_train, ratings_test)
    log.info("ranking metrics = %s " % rankings)
    wandb.log(data=rankings)
    # "precision", "map", "ndcg","auc"]
    return rankings["precision"]
