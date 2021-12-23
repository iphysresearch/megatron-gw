import sys
sys.path.append('/workspace/zhouy/megatron-gw/GWToolkit/')
from gwtoolkit.gw.ray import RayDatasetTorch, RayDataset, ray
import msgpack_numpy as m
m.patch()               # Important line to monkey-patch for numpy support!

import redis
import numpy as np

connection_pool = redis.ConnectionPool(host='localhost', port=5153, db=0, decode_responses=False)
r = redis.Redis(connection_pool=connection_pool)

def toRedis(value,name):
    """Store given Numpy array 'value' in Redis under key 'name'"""
    r.set(name,m.packb(value))
    return

from torch.utils.data import DataLoader
# import ray

import numpy as np


def update_level(i):
    if i%100==0:
        return 4
    elif i%50==0:
        return 3
    elif i%10==0:
        return 2
    else:
        return 1
    # np.array([update_level(i) for i in range(1000)])


batch_size = 1
num_dataset = 1
num_range = batch_size//num_dataset
num_repeat = 20

datasets = RayDatasetTorch.remote(num_dataset=num_dataset)

index = 0

while True:
    index += 1
    level = update_level(index)
    pipeline = datasets.pipeline.remote(num_range, num_repeat, batch_size, level=level)
    iteration = iter(ray.get(pipeline))
    for i, _ in enumerate(range(num_repeat)):
    # for i, (data, signal, noise, params) in enumerate(ray.get(pipeline)):
        (data, signal, params) = next(iteration)
        toRedis(data, f'data_{index}_{i}')
        toRedis(signal, f'signal_{index}_{i}')
        toRedis(params, f'params_{index}_{i}')
    if index * 20 == 1000:
        print('1000!===='*50)
        break

ray.shutdown()
