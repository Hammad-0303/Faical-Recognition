import torch

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
    #For hard positive distance
    mask.fill_diagonal_(False)
    hard_pos_dist = torch.where(mask, dist, torch.tensor(0.0, device=dist.device))
    hard_pos_dist = torch.max(hard_pos_dist, dim=1).values
    #For hard negative distance
    mask.fill_diagonal_(True)
    hard_neg_dist = torch.where(~mask, dist, torch.tensor(float('inf'), device=dist.device))
    hard_neg_dist = torch.min(hard_neg_dist, dim=1).values
    #Ensure that if now hard negative distance is found, it is replaced with 0
    hard_neg_dist = torch.where(hard_neg_dist == float('inf'), torch.tensor(0.0, device=dist.device), hard_neg_dist)

    #If there are not valid positives or valid negatives, then return the triplet loss as 0
    if torch.all(hard_pos_dist == 0) or torch.all(hard_neg_dist == 0):
        return torch.tensor(0.0, device=dist.device)

    #Compute the hard loss
    loss = torch.mean(torch.clamp(hard_pos_dist - hard_neg_dist + margin, min=0.0))
    return loss
