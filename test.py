#Testing all classes
from Helper_functions import LFWDataset
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import gc

#Test LFWDataset
root_dir = 'Data/lfw-deepfunneled/lfw-deepfunneled'
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
train_dataset = LFWDataset(root_dir, split = 'train', transform = transform)
val_dataset = LFWDataset(root_dir, split = 'val', transform = transform)
test_dataset = LFWDataset(root_dir, split = 'test', transform = transform)
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = True)

# Example: iterate through batches
for batch_idx, (anchors, positives, negatives) in enumerate(train_loader):
    print(batch_idx, anchors.shape, positives.shape, negatives.shape)
    if batch_idx == 2:
        break

