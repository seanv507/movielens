#!/usr/bin/env python
import os
from dask.distributed import wait
import coiled
import optuna
from optuna.integration.dask import DaskStorage
from optuna.trial import TrialState


import wandb

wandbc=None

def create_study():
#    create_software_environment()
    global wandbc
    wandbkey = os.environ["WANDB_API_KEY"]


    

    # client = cluster.get_client()
    
    n_trials = 10

    study = optuna.create_study(direction='minimize', 
                                load_if_exists=True)

    wandb_kwargs = {"project": "test_optuna_01"}
    wandbc = optuna.integration.WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)
    @wandbc.track_in_wandb() 
    def objective(trial):
        CFG = {"lr": 0.001, "num_epochs": 100}
        CFG = adjust_CFG(CFG, trial)
        wandb.config = {key:value for key, value in CFG.items()}
        x=10
        for epoch in range(CFG['num_epochs']):
            x -= 2 *CFG["lr"] * x
            epoch_val_loss = (x**2)
            trial.report(epoch_val_loss, epoch)
            wandb.log({"val_loss": epoch_val_loss, })
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    

        wandb.finish()
        return epoch_val_loss
    
    study.optimize(objective, n_trials=n_trials, callbacks=[wandbc])
    
    analyse_results(study)



def adjust_CFG(CFG, trial):
    CFG["lr"] = trial.suggest_categorical("learning rate", [0.1, 0.01, 0.001, 0.0001])
    return CFG



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
