import torch
from tqdm import tqdm
from dataset import NDCME


data = NDCME('./datasets/ndc-me/roi', './datasets/ndc-me/audio', 'train')

total_RGB_mean = torch.tensor((0.))
total_RGB_std = torch.tensor((0.))
for (d, a, l) in tqdm(iter(data)):
    total_RGB_mean += (a.mean())
    total_RGB_std += (a.std())

print(total_RGB_mean/len(data))
print(total_RGB_std/len(data))

