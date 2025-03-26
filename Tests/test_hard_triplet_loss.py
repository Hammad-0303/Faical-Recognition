import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import gc

def hard_triplet_loss(embeddings, labels, margin = 0.2):
    #Computes hard triplet loss
    #Parameters:
    #embeddings: Embeddings of images size (batch_size, embedding_size)
    #labels: Labels of images size (batch_size)
    #margin: Margin for triplet loss
    #Returns:
    #Triplet loss
    dist = torch.cdist(embeddings, embeddings, p=2)
    mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
    print(f'Mask: {mask}')
    #For hard positive distance
    mask.fill_diagonal_(False)
    print(f'Mask with false diagonal: {mask}')
    hard_pos_dist = torch.where(mask, dist, torch.tensor(0.0, device=dist.device))
    print(f'Positive distances: {hard_pos_dist}')
    hard_pos_dist = torch.max(hard_pos_dist, dim=1).values
    print(f'Hard positive distances: {hard_pos_dist}')
    #For hard negative distance
    mask.fill_diagonal_(True)
    print(f'Mask with true diagonal: {mask}')
    hard_neg_dist = torch.where(~mask, dist, torch.tensor(float('inf'), device=dist.device))
    print(f'Negative distances: {hard_neg_dist}')
    hard_neg_dist = torch.min(hard_neg_dist, dim=1).values
    print(f'Hard negative distances: {hard_neg_dist}')
    #Ensure that if now hard negative distance is found, it is replaced with 0
    hard_neg_dist = torch.where(hard_neg_dist == float('inf'), torch.tensor(0.0, device=dist.device), hard_neg_dist)
    print(f'Hard negative distances after replacing inf: {hard_neg_dist}')

    #If there are not valid positives or valid negatives, then return the triplet loss as 0
    if torch.all(hard_pos_dist == 0) or torch.all(hard_neg_dist == 0):
        return torch.tensor(0.0, device=dist.device)

    #Compute the hard loss
    loss = torch.mean(torch.clamp(hard_pos_dist - hard_neg_dist + margin, min=0.0))
    return loss

#Tests
# Small batch with clear positive and negative pairs
embeddings = torch.tensor([[1.0, 2.0], [1.1, 2.1], [5.0, 5.0], [5.1, 5.1]])
labels = torch.tensor([0, 0, 1, 1])  # First two belong to class 0, last two to class 1

loss = hard_triplet_loss(embeddings, labels)
print("Test Case 1 - Basic Case Loss:", loss.item())

# All samples belong to the same class, so no valid negatives
embeddings = torch.tensor([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3]])
labels = torch.tensor([0, 0, 0, 0])  # All belong to class 0

loss = hard_triplet_loss(embeddings, labels)
print("Test Case 2 - Only One Class Loss:", loss.item())

# Every embedding has a unique class
embeddings = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
labels = torch.tensor([0, 1, 2, 3])  # All different

loss = hard_triplet_loss(embeddings, labels)
print("Test Case 3 - All Unique Classes Loss:", loss.item())

# More realistic case with multiple classes
embeddings = torch.randn(10, 128)  # 10 samples with 128-d embeddings
labels = torch.tensor([0, 1, 1, 2, 2, 2, 3, 3, 4, 4])  # Multiple classes

loss = hard_triplet_loss(embeddings, labels)
print("Test Case 4 - Large Batch Loss:", loss.item())

embeddings = torch.tensor([
    [1.0, 1.0], [1.1, 1.1],  # Class 0 (very close)
    [5.0, 5.0], [10.0, 10.0]  # Class 1 (far apart)
])
labels = torch.tensor([0, 0, 1, 1])

loss = hard_triplet_loss(embeddings, labels)
print("Test Case 5 - Positives Closer than Negatives Loss:", loss.item())