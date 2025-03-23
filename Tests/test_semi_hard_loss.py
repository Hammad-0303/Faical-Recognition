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
#Modified function with print statements to understand the working of the function
def semi_hard_triplet_loss(anchor, positive, negative, margin = 0.2):
    #Computes semi hard triplet loss and if no triplets are found
    #returns to the loss of random triplets
    #Parameters:
    #anchor: Embedding of anchor image
    #positive: Embedding of positive image  
    #negative: Embedding of negative image
    #margin: Margin for triplet loss
    #Returns:
    #Triplet loss
    pos_dist = F.pairwise_distance(anchor, positive, p=2)
    neg_dist = F.pairwise_distance(anchor, negative, p=2)
    print(f"Positive distance: {pos_dist}, Negative distance: {neg_dist}")
    mask = torch.logical_and(pos_dist < neg_dist , neg_dist < pos_dist + margin)
    print(f"mask: {mask}")
    print(f"Number of semi-hard triplets: {mask.sum()}")
    if mask.any():
        loss = torch.mean(torch.clamp(pos_dist[mask] - neg_dist[mask] + margin, min=0.0))
        print(f"Normal loss: {loss:.4f}")
    else:
        loss = torch.mean(torch.clamp(pos_dist - neg_dist + margin, min=0.0))
        print(f"Random loss: {loss:.4f}")
    return loss


# Generate random embeddings for a batch of 128 samples with 256-d features
anchor = torch.randn(128, 256)
positive = anchor + torch.randn(128, 256) * 0.1  # Slightly different from anchor
negative = anchor + torch.randn(128, 256) * 0.3  # Completely random

loss = semi_hard_triplet_loss(anchor, positive, negative, margin=0.2)
print(f"Loss on large batch: {loss.item()}")