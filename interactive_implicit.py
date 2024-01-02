#%%
import time

import numpy as np


from implicit.als import AlternatingLeastSquares
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



#%%

#%%
def make_implicit_movielens(ratings, min_rating):
    # remove things < min_rating, and convert to implicit dataset
    # by considering ratings as a binary preference only
    ratings.data[ratings.data < min_rating] = 0
    ratings.eliminate_zeros()
    ratings.data = np.ones(len(ratings.data))
    return ratings    


#%%
# read in the input data file
variant="1m"
min_rating=4.0
start = time.time()
titles, ratings = get_movielens(variant)
print("read data file in %s", time.time() - start)
ratings = make_implicit_movielens(ratings, min_rating)
assert len(titles)==ratings.shape[0]
# ratings is film rows x user columns

#%%

#%%
model_kwargs = {}
model_kwargs["use_gpu"] = False
model_kwargs["use_cg"] = True
model_kwargs["factors"] = 100
model_kwargs["regularization"] = 1e-2
model_kwargs["iterations"] = 16
model_kwargs["calculate_training_loss"] = True
use_BM25 = False




    # generate a recommender model based off the input params
model =  AlternatingLeastSquares(
    **model_kwargs,
)


print("has cuda %s" % HAS_CUDA)


#%%

if use_BM25:
    # lets weight these models by bm25weight.
    print("weighting matrix by bm25_weight")
    ratings = (bm25_weight(ratings, B=0.9) * 5) #converts to coo
    ratings = ratings.tocsr()

user_ratings = ratings.T
assert user_ratings.shape[1] == len(titles)
ratings_train, ratings_test = leave_k_out_split(user_ratings, K=5, train_only_size=0.0)
#%%


# train the model
start = time.time()
model.fit(user_ratings)
print("trained model in %s", time.time() - start)
#%%
rankings = ranking_metrics_at_k(model, ratings_train, ratings_test)
print("ranking metrics = %s " % rankings)
# "precision", "map", "ndcg","auc"]




# %%
