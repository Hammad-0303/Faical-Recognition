import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import torchvision.transforms as transforms
import gc
from collections import defaultdict, deque
import heapq


class LFWDataset(Dataset):
    def __init__(self, root_dir, split = 'train', transform = None, mode = 'triplet'):
        #Splits the LFW dataset into train, validation, and test sets
        #Generates triplets (Anchor, Positive, Negative) dynamically
        #Parameters:
        #root_dir: Path to LFW dataset
        #split: 'train', 'val', or 'test'
        #transform: Image transformation
        #mode: 'triplet' or 'hard' for triplet loss or hard triplet loss
        random.seed(42) #For reproducibility
        torch.manual_seed(42) #For reproducibility
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.mode = mode
        #Collect all idenitites with images and store them in a dictionary
        person_images = {person : [os.path.join(root_dir, person, img) for img in os.listdir(os.path.join(root_dir,person)) if img.endswith('.jpg')]
                         for person in os.listdir(root_dir) 
                         if os.path.isdir(os.path.join(root_dir,person)) and os.listdir(os.path.join(root_dir,person))}
        
        #List of all the people
        all_people = list(person_images.keys())
        # Shuffle and split dataset (80% train, 10% val, 10% test)
        random.shuffle(all_people)
        total_people = len(all_people)
        split_dict = {'train': all_people[:int(0.8*total_people)],
                      'val': all_people[int(0.8*total_people):int(0.9*total_people)],
                      'test': all_people[int(0.9*total_people):]}
        
        #Select the split
        self.all_identities = {person : person_images[person] for person in split_dict[split]}

        if mode == 'triplet':
            self.valid_identities = {person : person_images[person] for person in split_dict[split] if len(person_images[person])>1}

            #Number of pairs
            self.num_pairs = sum((len(images)*(len(images)-1))//2 for images in self.valid_identities.values())
        
        elif mode == 'hard':
            self.person_to_id = {person : idx for idx, person in enumerate(self.all_identities.keys())} #Assign a unique id to each person
            self.id_image_pair = [(self.person_to_id[person], img) for person, images in self.all_identities.items() for img in images] #Assign person id to each image
            del self.person_to_id #Free memory

        #Free memory
        del person_images, all_people, split_dict
        gc.collect()
    
    def __len__(self):
        if self.mode == 'triplet':
            return min(150000, max(1000,3*(self.num_pairs))) #Ensuring that the number of triplets is at least 1000
        elif self.mode == 'hard':
            return len(self.id_image_pair) #Number of images in the dataset
    
    def __getitem__(self, idx):

        if self.mode == 'triplet':
            #Randomly select a person from anchor and positive pairs
            anchor_person = random.choice(list(self.valid_identities.keys()))
            anchor_images = self.valid_identities[anchor_person]
            anchor_path, positive_path = random.sample(anchor_images, 2)

            #Randomly select a negative person
            negative_persons = [p for p in self.all_identities.keys() if p!=anchor_person]
            negative_person = random.choice(negative_persons)
            negative_path = random.choice(self.all_identities[negative_person])

            #Load images
            anchor = Image.open(anchor_path).convert('RGB')
            positive = Image.open(positive_path).convert('RGB')
            negative = Image.open(negative_path).convert('RGB')

            if self.transform:
                anchor = self.transform(anchor)
                positive = self.transform(positive)
                negative = self.transform(negative)

            return anchor, positive, negative
        
        elif self.mode == 'hard':
            #Sort image by the usage count
            selected_personid, selected_image = self.id_image_pair[idx] 
            #Load image
            image = Image.open(selected_image).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, selected_personid
        
class BalancedHardTripletSampler(Sampler):
    def __init__(self, dataset, batch_size = 32 ,min_samples_per_class=2):
        #Sampler for balanced hard triplet loss ensuring that we have atleast a triplet pair in a batch
        #Parameters:
        #dataset: Dataset to sample from
        #batch_size: Batch size
        #min_samples_per_class: Minimum number of samples per class
        random.seed(42) #For reproducibility
        torch.manual_seed(42)
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_samples_per_class = min_samples_per_class
        self.person_to_indices = defaultdict(list)
        #Create a dictionary which groups indices with their respective person id
        for idx, (pid, _) in enumerate(dataset.id_image_pair):
            self.person_to_indices[pid].append(idx)
        
        self.valid_pair = {pid: indices for pid, indices in self.person_to_indices.items() 
                           if len(indices) >= min_samples_per_class}  #All person index pair with atleast 2 images
        self.all_classes = list(self.person_to_indices.keys())  #All people

        if not self.valid_pair:
            raise ValueError("No valid classes with sufficient samples found in the dataset.")
        
        self.image_count = {idx: 0 for idx in range(len(self.dataset.id_image_pair))}  #Count of images used in the batch
        self.class_count = {pid: 0 for pid in self.all_classes} #Count of classes used in the batch
        self.class_heap = [(0, pid) for pid in self.all_classes]
        heapq.heapify(self.class_heap)  #Heapify the class heap for efficient access to the least used classes
        self.valid_class = list(self.valid_pair.keys())  #List of valid classes 
    def least_used_items(self, count, items):
        #Selects least used image using a min heap for efficient access
        #Parameters:
        #count: Number of least used items to select
        #items: List of items to select from
        #Returns:
        if count + int(0.3*count) > len(items):
            sample_size = count
        else:
            sample_size = count + int(0.3*count)
        n_smallest = heapq.nsmallest(sample_size, items, key = lambda idx: self.image_count[idx])
        return random.sample(n_smallest, count)  #Randomly select from the least used items
    
    def __iter__(self):
        #Iterates through the dataset and returns indices for the batch
        #Returns:
        #Indices for the batch
        num_batches = len(self)
        for _ in range(num_batches):
            batch_indices = []
            #Select a anchor and positive pair
            anchor_class = random.choice(self.valid_class[:10]) #Randomly select a class from the first 10 classes in the queue
            self.valid_class.remove(anchor_class)  #Remove the selected class from the valid class list
            self.valid_class.append(anchor_class)  #Add the selected class to the end of the list
            selected_anchor_indices = self.least_used_items(2, self.valid_pair[anchor_class])  #Select least used images for anchor and positive
            batch_indices.extend(selected_anchor_indices)  #Add anchor and positive indices to the batch

            #Select a negative class
            negative_class = None
            temp_heap = []
            while self.class_heap:
                count, person = heapq.heappop(self.class_heap)  #Select the least used class
                if person != anchor_class:
                    if random.random() < 0.3:
                        #Adds a 30% random chance to skip the class
                        temp_heap.append((count, person))  #Keep track of popped items to restore later
                        continue  #Skip this class
                    negative_class = person
                    break
                else:
                    #Keep track of popped items to restore later
                    temp_heap.append((count, person))
            
            #Restore the popped items
            for item in temp_heap:
                heapq.heappush(self.class_heap, item)
            
            if not negative_class:
                negative_class = random.choice([c for c in self.all_classes if c != anchor_class])  #Fallback to random selection if all classes are exhausted
            
            #Restore and update the count in heap
            self.class_count[negative_class] += 1
            heapq.heappush(self.class_heap, (self.class_count[negative_class], negative_class))

            #Select negative index
            selected_negative_indices = self.least_used_items(1, self.person_to_indices[negative_class])  #Select least used image for negative
            batch_indices.extend(selected_negative_indices)  #Add negative indices to the batch

            #Fill remainig indices
            remaining_slots = self.batch_size - len(batch_indices)
            if remaining_slots > 0:
                #Select least used images from the valid pair
                remaining_indices = [idx for idx in range(len(self.dataset.id_image_pair)) if idx not in batch_indices]
                if remaining_indices: 
                    remaining_indices = self.least_used_items(remaining_slots, remaining_indices)  #Select least used images for the remaining slots
                    batch_indices.extend(remaining_indices)  #Add remaining indices to the batch
            
            #Update the image count for the batch
            for idx in batch_indices:
                self.image_count[idx] += 1
            
            #Yield the batch indices
            yield batch_indices[:self.batch_size]  #Ensure the batch size is maintained
        
    def __len__(self):
        #Returns the number of batches in the dataset
        #Returns:
        #Number of batches in the dataset
        return (len(self.dataset.id_image_pair)+self.batch_size-1) // self.batch_size



#Importing weights from resnet50 and modifying the last layer    
class FaceEmbeddingModel(nn.Module):
    def __init__(self, drop_prob = 0.2):
        super().__init__()
        resnet50 = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet50.children())[:-1]) #Removing the last layer
        self.fc1 = nn.Linear(2048, 512)
        self.dropout = nn.Dropout(drop_prob)
        self.fc2 = nn.Linear(512, 128)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
def save_model(model, optimizer,epoch,loss, path = "face_embedding_model.pth"):
    # Saves the model's state, optimizer state, and training metadata.
    
    # Parameters:
    #     model (nn.Module): The trained model.
    #     optimizer (torch.optim.Optimizer): The optimizer used during training.
    #     epoch (int): Current epoch number.
    #     loss (float): Training loss at the time of saving.
    #     path (str): File path to save the model.
    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': loss}
    torch.save(checkpoint, path)
    print(f"Model saved at {path} during epoch {epoch} with loss {loss:.4f}")

def load_model(model, optimizer, path = "face_embedding_model.pth"):
    # Loads the model's state, optimizer state, and training metadata.
    
    # Parameters:
    #     model (nn.Module): The model to load the state_dict into.
    #     optimizer (torch.optim.Optimizer): The optimizer to load the state_dict into.
    #     path (str): File path to load the model.
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']
   
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

    


    





            







        
        
