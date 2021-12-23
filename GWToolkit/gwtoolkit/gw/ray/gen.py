import sys
sys.path.append('/workspace/zhouy/megatron-gw/GWToolkit/')
from gwtoolkit.gw.ray import RayDatasetTorch, RayDataset, ray
from gwtoolkit.redis.utils import toRedis, is_exist_Redis

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
num_repeat = 100

datasets = RayDatasetTorch.remote(num_dataset=num_dataset)

index = 0

while True:
    try:
        index += 1
        print(f'Runing on index={index}.')

        level = update_level(index)
        pipeline = datasets.pipeline.remote(num_range, num_repeat, batch_size, level=level)
        iteration = iter(ray.get(pipeline))
        for i, _ in enumerate(range(num_repeat)):
        # for i, (data, signal, noise, params) in enumerate(ray.get(pipeline)):
        if is_exist_Redis(f'data_{index}_{i}'):
            seed = np.random.rand()
            toRedis(seed, f'seed_data_{index}_{i}')
            toRedis(seed, f'seed_signal_{index}_{i}')
            toRedis(seed, f'seed_params_{index}_{i}')      
            
            (data, signal, params) = next(iteration)
            toRedis(data, f'data_{index}_{i}')
            toRedis(signal, f'signal_{index}_{i}')
            toRedis(params, f'params_{index}_{i}')

      
    except KeyboardInterrupt:
        print('finish!')
        break

ray.shutdown()