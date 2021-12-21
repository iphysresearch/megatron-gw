import sys
# sys.path.append('/workspace/zhaoty/GWToolkit/')
from GWToolkit.gwtoolkit.gw.ray import RayDatasetTorch, ray

def update_level(i):
    if i%100==4:
        return 4
    elif i%50==0:
        return 3
    elif i%10==0:
        return 2
    else:
        return 1

batch_size = 128
num_dataset = 32
num_range = batch_size//num_dataset
num_repeat = 50
datasets = RayDatasetTorch.remote(num_dataset=num_dataset)
index=0
level = update_level(index)  # index = iteration
pipeline = datasets.pipeline.remote(num_range, num_repeat, batch_size, level=level)
train_ds = ray.get(pipeline)