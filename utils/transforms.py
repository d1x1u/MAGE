import numpy as np
import torch
from torch import nn
from torchvision import transforms


class RandomRotate(nn.Module):
    def __init__(self):
        super().__init__()
        self.times = np.random.randint(low=0, high=4)
    
    def forward(self, x):
        out = transforms.functional.rotate(x, 90*self.times)
        return out

class RandomPointErasure(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        prob = torch.empty(patch_size, patch_size).uniform_(0.9, 1)
        self.mask = torch.bernoulli(input=prob)
        self.mask[patch_size//2, patch_size//2] = 1.0
    
    def forward(self, x):
        out = x * self.mask.to(x.device)
        return out

def get_train_transforms(patch_size):
    data_transforms = nn.Sequential(
        RandomRotate(),
        transforms.RandomResizedCrop(size=patch_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomErasing()
    )
    return data_transforms
