import sys
sys.path.append('/workspace/zhouy/megatron-gw/GWToolkit/')
from gwtoolkit.gw.ray import RayDatasetTorch, RayDataset, ray
from gwtoolkit.redis.utils import toRedis, is_exist_Redis

from torch.utils.data import DataLoader
from tqdm import tqdm

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
num_dataset = 32 if batch_size >= 32 else batch_size
num_range = batch_size//num_dataset
num_repeat = 4

datasets = RayDatasetTorch.remote(num_dataset=num_dataset)
start = 6_0000 // num_repeat
end = 10_0000 // num_repeat

# for index in tqdm(range(start//num_repeat, end//num_repeat)):
index = start
while True:
    try:
        level = update_level(index)
        pipeline = datasets.pipeline.remote(num_range, num_repeat, batch_size, level=level)
        iteration = iter(ray.get(pipeline))
        for i, _ in enumerate(range(num_repeat)):
            # if is_exist_Redis(f'data_{index}_{i}'):
            #     continue
            seed = np.random.rand()

            (data, signal, params) = next(iteration)
            toRedis(data, f'data_{index}_{i}')
            toRedis(seed, f'seed_data_{index}_{i}')

            toRedis(signal, f'signal_{index}_{i}')
            toRedis(seed, f'seed_signal_{index}_{i}')

            toRedis(params, f'params_{index}_{i}')
            toRedis(seed, f'seed_params_{index}_{i}')
        index += 1
        if index == end - 1:
            index = start
    except KeyboardInterrupt:
        break
ray.shutdown()
