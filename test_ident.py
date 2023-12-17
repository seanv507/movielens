import coiled
# from distributed import wait
# from models import process
print(coiled.utils.get_aws_identity())
# df = process.load_movielens_1m()
# cluster = coiled.Cluster(
#         name="pytorch",
#         #software="pytorch",
#         n_workers=2,
#         worker_gpu=1,
#         scheduler_gpu=0,
#         # launch one task per worker to avoid oversaturating the GPU
#         worker_options={"nthreads": 1},
# )

# client = cluster.get_client()

# val = client.submit(coiled.utils.get_aws_identity).result()
# print(val)
import pandas as pd
df
print(coiled.list_instance_types("aws", min_gpus=1))
