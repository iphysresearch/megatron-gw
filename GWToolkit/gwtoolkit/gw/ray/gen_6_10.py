import sys
sys.path.append('/workspace/zhouy/megatron-gw/GWToolkit/')
from gwtoolkit.gw.ray import RayDatasetTorch, RayDataset, ray
from gwtoolkit.redis.utils import toRedis, is_exist_Redis
from gwtoolkit.redis import DatasetTorchRealEvent

from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np


def update_level(i):
    return 0
    if i%100==0:
        return 4
    elif i%50==0:
        return 3
    elif i%10==0:
        return 2
    else:
        return 1
    # np.array([update_level(i) for i in range(1000)])


batch_size = 32
num_dataset = 32 if batch_size >= 32 else batch_size
num_range = batch_size//num_dataset
num_repeat = 2

datasets = RayDatasetTorch.remote(num_dataset=num_dataset)
start = 6_0000 // num_repeat // batch_size
end = 10_0000 // num_repeat // batch_size


d = DatasetTorchRealEvent()
t = np.arange(d.start_time+d.duration_long//2-d.duration//2,
              d.start_time+d.duration_long//2-d.duration//2+d.duration, 1/d.sampling_frequency)


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

            (data, signal, params) = next(iteration)
            for j in range(batch_size):
                seed = np.random.rand()
                toRedis(data[j:j+1], f'data_{index}_{j}_{i}')
                toRedis(seed, f'seed_data_{index}_{j}_{i}')

                toRedis(signal[j:j+1], f'signal_{index}_{j}_{i}')
                toRedis(seed, f'seed_signal_{index}_{j}_{i}')

                toRedis(params[j:j+1], f'params_{index}_{j}_{i}')
                toRedis(seed, f'seed_params_{index}_{j}_{i}')

                left = np.real(params[j, -5]) - 0.4
                right = np.real(params[j, -5]) + 0.1
                toRedis(np.array([[np.argmax(t > left), np.argmin(t < right)]], dtype=np.int32), f'mask_{index}_{j}_{i}')
                toRedis(seed, f'seed_mask_{index}_{j}_{i}')

        index += 1
        if index == end - 1:
            index = start
    except KeyboardInterrupt:
        break
ray.shutdown()
