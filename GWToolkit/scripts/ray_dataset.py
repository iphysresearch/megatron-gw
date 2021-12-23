import sys
sys.path.append('/workspace/zhouy/megatron-gw/GWToolkit/')
from gwtoolkit.gw.ray import RayDatasetTorch, RayDataset, ray
import time
import torch

def update_level(i):
    if i%100==4:
        return 4
    elif i%50==0:
        return 3
    elif i%10==0:
        return 2
    else:
        return 1
    # np.array([update_level(i) for i in range(1000)])
    
    
batch_size = 128
num_dataset = 32
num_range = batch_size//num_dataset
num_repeat = 50

# num_dozen= 16
# test = True
datasets = RayDatasetTorch.remote(num_dataset=num_dataset)



index = 0
while True:
    index += 1
    level = update_level(index)
    pipeline = datasets.pipeline.remote(num_range, num_repeat, batch_size, level=level)
    iteration = iter(ray.get(pipeline))
    start = time.time()
    for i, _ in enumerate(range(num_repeat)):
        end = time.time()
        (data, signal, params) = next(iteration)
        print(f'batch={i}, time: {end-start:.4f}sec', data.shape, signal.shape, params['geocent_time'][:2].tolist())
        start = end


input('Ctrl + C')
index = 0
while True:
    index += 1
    level = update_level(index)
    pipeline = datasets.pipeline.remote(num_range, num_repeat, batch_size, level=level)
    train_dataloader = iter(torch.utils.data.DataLoader(ray.get(pipeline), num_workers=0))
    start = time.time()
    for i, (data, signal, params) in enumerate(train_dataloader):
    # for i, (data, signal, noise, params) in enumerate(ray.get(pipeline)):
        end = time.time()
        # print(f'batch={i}, time: {end-start:.4f}sec', data.shape, signal.shape, noise.shape, params['geocent_time'][:2].tolist())
        print(f'batch={i}, time: {end-start:.4f}sec', data.shape, signal.shape, params['geocent_time'][:2].tolist())
        start = end

