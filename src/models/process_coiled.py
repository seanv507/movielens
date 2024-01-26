import os
import coiled
from optuna.trial import TrialState


def get_client():
    wandbkey = os.environ["WANDB_API_KEY"]
    cluster = coiled.Cluster(
        name="pytorch",
        software="pytorch",
        #package_sync=False,
        n_workers=3,
        worker_gpu=0,
        scheduler_gpu=0,
        # launch one task per worker to avoid oversaturating the GPU
        worker_options={"nthreads": 1},
        spot_policy="spot",
    )
    cluster.send_private_envs({"WANDB_API_KEY": wandbkey})
    client = cluster.get_client()
    return client

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
                "implicit",
                "dask",
                "distributed",
                "pynvml",
                "wandb",
                "optuna",
                "scikit-learn",
                "Cython",
            ],
        },
        pip=["fastfm","git+https://GIT_TOKEN@github.com/seanv507/movielens.git"],
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
