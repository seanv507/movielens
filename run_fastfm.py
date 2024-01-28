#!/usr/bin/env python
import os
from dask.distributed import wait
import optuna
from optuna.integration.dask import DaskStorage


from models import process_fastfm, process_coiled


# compute the AUC metric by repeatedly 
# randomly splitting the dataset into a 80% training set and a 20% test set.
# The final score is given averaging across 10 repetitions.

def create_study():
    process_coiled.create_software_environment()

    wandb_kwargs = {"project": "matrix_factorisation_movielens_test_04"}
    wandbc = optuna.integration.WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)
    @wandbc.track_in_wandb()
    def objective(trial):
        objective_value = process_fastfm.objective(trial)
        return objective_value

    client = process_coiled.get_client()

    n_trials = 400

    study = optuna.create_study(direction='maximize', 
                                storage=DaskStorage(client=client),
                                load_if_exists=True)
    # study.optimize(objective, n_trials=1, callbacks=[wandbc])
    # exit(0)
    jobs = [
        client.submit(study.optimize, objective, n_trials=1, callbacks=[wandbc],pure=False)
        for _ in range(n_trials)
    ]
    _ = wait(jobs)

    process_coiled.analyse_results(study)


if __name__ == "__main__":
    create_study()
