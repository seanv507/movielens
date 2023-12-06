#%%
import wandb
import models
import os
os.environ["AWS_PROFILE"]="sean"
#%%
%load_ext autoreload
%autoreload all
#%%
from models import process
#%%
#%%
# plt.figure(figsize=(4, 1.5))
# bins = np.arange(1, 6, 0.5) - 0.25
# plt.hist(df.rating.values, bins=bins)
# plt.xticks(np.arange(1, 5.01, 1))
# plt.title('Num Ratings vs Rating')
# plt.xlabel('Rating')
# plt.ylabel('Num Ratings')
# plt.grid()


# %%

# min_num_ratings = df.groupby(['userId'])['userId'].transform(len).min()
# print(f'Min # of ratings per user: {min_num_ratings}')
# print(f'Min/Max rating: {df.rating.min()} / {df.rating.max()}')
# print(f'df.shape: {df.shape}')
# df.head(3)
#%%
df = process.load_movielens_1m()
df = process.convert_categorical(df)
df, n_feats = process.make_embedding_indices(df)
#%%
df_train, df_val = process.split_train_test(df)
dl_train = process.make_dataset(df_train)
dl_val = process.make_dataset(df_val)

#%%

model_t = "FM"
CFG = {
        'model': model_t,
        'lr': 0.0001,
        'num_epochs': 1000,
        'weight_decay': 0.4,
        'emb_dim': 50,
        'bias': True,
        'init_std': 0.01,
        'seed': 123,
    }


model_setup = process.create_model_setup(CFG, n_feats)
#%%
epoch_train_losses, epoch_val_losses = [], []
#%%
wandb.init(project="matrix_fact_0", config = CFG)
for i in range(CFG['num_epochs']):
    mdl, epoch_train_loss, epoch_val_loss = process.train_model(**model_setup, dl_train=dl_train, dl_val=dl_val)
    s = (f'Epoch: {i}, Train Loss: {epoch_train_loss:0.2f}, '
        f'Val Loss: {epoch_val_loss:0.2f}'
    )    
    print(s)
    epoch_train_losses.append(epoch_train_loss)
    epoch_val_losses.append(epoch_val_loss)

wandb.finish()


# %%
