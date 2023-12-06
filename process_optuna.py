#!/usr/bin/env python
import os
from dask.distributed import wait
import coiled
import optuna
from optuna.integration.dask import DaskStorage
from optuna.trial import TrialState


import wandb
from models import process


def create_study():
#    create_software_environment()
    wandbkey = os.environ["WANDB_API_KEY"]
    wandb_kwargs = {"project": "my-project"}
    wandbc = optuna.integration.WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)

    # client = cluster.get_client()
    cluster = coiled.Cluster(
        name="pytorch",
        #software="pytorch",
        package_sync=True,
        n_workers=2,
        worker_gpu=1,
        scheduler_gpu=0,
        # launch one task per worker to avoid oversaturating the GPU
        worker_options={"nthreads": 1},
        #spot_policy="spot",
    )
    cluster.send_private_envs({"WANDB_API_KEY": wandbkey})
    client = cluster.get_client()

    n_trials = 10

    study = optuna.create_study(direction='minimize', 
                                storage=DaskStorage(client=client),
                                load_if_exists=True)

    jobs = [
        client.submit(study.optimize, objective, n_trials=1, pure=False, callbacks=[wandbc])
        for _ in range(n_trials)
    ]
    # _ = wait(jobs)

    # analyse_results(study)

def objective(trial):
    df = process.get_movielens_1m()
    n_feats = process.get_n_feats(df)

    df_train, df_val = process.split_train_test(df)
    dl_train = process.make_dataset(df_train)
    dl_val = process.make_dataset(df_val)


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
    CFG = adjust_CFG(CFG, trial)

    model_setup = process.create_model_setup(CFG, n_feats)
    epoch_train_losses, epoch_val_losses = [], []
    wandb.init(project="matrix_fact_0", config = CFG)
    for epoch in range(CFG['num_epochs']):
        mdl, epoch_train_loss, epoch_val_loss = process.train_model(**model_setup, dl_train=dl_train, dl_val=dl_val)
        s = (f'Epoch: {epoch}, Train Loss: {epoch_train_loss:0.2f}, '
            f'Val Loss: {epoch_val_loss:0.2f}'
        )    
        print(s)
        epoch_train_losses.append(epoch_train_loss)
        epoch_val_losses.append(epoch_val_loss)
        trial.report(epoch_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    

    # wandb.finish() unneeded
    return epoch_val_loss


def adjust_CFG(CFG, trial):
    CFG["emb_dim"]  = trial.suggest_categorical("embedding dimension", [1,2,4,8,16,32,64])
    CFG["weight_decay"] = trial.suggest_categorical("weight decay", [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001])
    CFG["lr"] = trial.suggest_categorical("learning rate", [0.1, 0.01, 0.001, 0.0001])
    CFG["init_std"] = trial.suggest_categorical("init_std", [0.01, 0.1])
    return CFG

def create_software_environment():
    coiled.create_software_environment(
        name="pytorch",
        conda={
            "channels": ["pytorch", "nvidia", "conda-forge", "defaults"],
            "dependencies": [
                "python=3.11",
                "coiled",
                "pytorch",
                "torchvision",
                "torchaudio",
                "cudatoolkit",
                "dask",
                "distributed",
                "pynvml",
                "wandb",
                "optuna",
            ],
        },
        gpu_enabled=True,
        region_name="eu-central-1"
    )



def analyse_results(study):
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    create_study()
