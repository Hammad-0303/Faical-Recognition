#Testing all classes

from Helper_functions import LFWDataset
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
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


# Define the inverse normalization transform
inv_normalize = transforms.Compose([
    transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])])

# Function to denormalize and convert tensor to a numpy image
def denormalize_and_convert(tensor):
    tensor = inv_normalize(tensor)  # Undo normalization
    tensor = torch.clamp(tensor, 0, 1)  # Ensure values are between 0 and 1
    return tensor.permute(1, 2, 0).numpy()  # Convert to HWC format for display

# Load dataset
dataset = LFWDataset(root_dir, split="train", transform=transform)
# Create DataLoader
dataloader = DataLoader(dataset, batch_size=6, shuffle=True)

# Get one batch
anchors, positives, negatives = next(iter(dataloader))

# Plot images
fig, axes = plt.subplots(6, 3, figsize=(10, 15))

for i in range(6):
    axes[i, 0].imshow(denormalize_and_convert(anchors[i]))
    axes[i, 0].set_title("Anchor")
    axes[i, 1].imshow(denormalize_and_convert(positives[i]))
    axes[i, 1].set_title("Positive")
    axes[i, 2].imshow(denormalize_and_convert(negatives[i]))
    axes[i, 2].set_title("Negative")

    for j in range(3):
        axes[i, j].axis("off")  # Hide axis

plt.tight_layout()
plt.show()

