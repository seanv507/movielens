#!/usr/bin/env python
import os
from dask.distributed import wait
import coiled
import optuna
from optuna.integration.dask import DaskStorage


import wandb
from models import process_implicit, process_coiled



def create_study():
#    create_software_environment()

    wandb_kwargs = {"project": "matrix_factorisation_movielens_test_01"}
    wandbc = optuna.integration.WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)
    @wandbc.track_in_wandb()
    def objective(trial):
        objective_value = process_implicit.objective(trial)
        return objective_value

    client = process_coiled.get_client()

    n_trials = 100

    study = optuna.create_study(direction='maximize', 
                                storage=DaskStorage(client=client),
                                load_if_exists=True)
    

    jobs = [
        client.submit(study.optimize, objective, n_trials=1, pure=False, callbacks=[wandbc])
        for _ in range(n_trials)
    ]
    _ = wait(jobs)

    process_coiled.analyse_results(study)


def adjust_CFG(CFG, trial):
    CFG["emb_dim"]  = trial.suggest_categorical("embedding dimension", [4])#[1,2,4,8,16,32,64])
    CFG["weight_decay"] = trial.suggest_categorical("weight decay", [0.1, 0.01])
    CFG["lr"] = trial.suggest_categorical("learning rate", [0.1, 0.01, 0.002, 0.001, 0.0001])
    CFG["init_std"] = trial.suggest_categorical("init_std", [0.01])
    return CFG




if __name__ == "__main__":
    create_study()
