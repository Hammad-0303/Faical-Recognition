import torch
import torch.nn.functional as F

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
    mask = torch.logical_and(pos_dist < neg_dist , neg_dist < pos_dist + margin)
    if mask.any():
        loss = torch.mean(torch.clamp(pos_dist[mask] - neg_dist[mask] + margin, min=0.0))
    else:
        loss = torch.mean(torch.clamp(pos_dist - neg_dist + margin, min=0.0))
    return loss