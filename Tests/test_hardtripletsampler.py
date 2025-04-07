from Helper_functions import *
import os
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import time
import collections
from collections import Counter
import random
import heapq
from collections import defaultdict
from torch.utils.data import Sampler

class BalancedHardTripletSamplerTest(Sampler):
    def __init__(self, dataset, batch_size=32, min_samples_per_class=2):
        # Sampler for balanced hard triplet loss ensuring that we have at least a triplet pair in a batch.
        # Parameters:
        # dataset: Dataset to sample from
        # batch_size: Batch size
        # min_samples_per_class: Minimum number of samples per class
        random.seed(42)  # For reproducibility
        torch.manual_seed(42)
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_samples_per_class = min_samples_per_class
        self.person_to_indices = defaultdict(list)
        
        # Create a dictionary that groups indices with their respective person id
        for idx, (pid, _) in enumerate(dataset.id_image_pair):
            self.person_to_indices[pid].append(idx)
        
        # Filter out persons with fewer images than the minimum required
        self.valid_pair = {pid: indices for pid, indices in self.person_to_indices.items() 
                           if len(indices) >= min_samples_per_class}
        self.all_classes = list(self.person_to_indices.keys())  # List of all unique classes (people)
        
        if not self.valid_pair:
            raise ValueError("No valid classes with sufficient samples found in the dataset.")
        
        self.image_count = {idx: 0 for idx in range(len(self.dataset.id_image_pair))}  # Count of images used in the batch
        self.class_count = {pid: 0 for pid in self.all_classes}  # Count of classes used in the batch
        self.class_heap = [(0, pid) for pid in self.all_classes]
        heapq.heapify(self.class_heap)  # Heapify the class heap for efficient access to the least used classes
        self.valid_class = list(self.valid_pair.keys())  # List of valid classes with sufficient samples
    
    def least_used_items(self, count, items):
        # Selects least used images using a min-heap for efficient access
        # Parameters:
        # count: Number of least used items to select
        # items: List of items (indices) to select from
        # Returns a list of randomly selected least used items from the provided list.
        
        if count + int(0.3*count) > len(items):
            sample_size = count
        else:
            sample_size = count + int(0.3*count)
        
        n_smallest = heapq.nsmallest(sample_size, items, key=lambda idx: self.image_count[idx])
        selected_items = random.sample(n_smallest, count)
        print(f'sample_size: {sample_size}')
        print(f"Selected {count} least used items: {selected_items}")
        return selected_items  # Randomly select from the least used items
    
    def __iter__(self):
        # Iterates through the dataset and returns indices for the batch
        # Returns:
        # Indices for the batch
        num_batches = len(self)
        
        for batch_idx in range(num_batches):
            print(f"Batch {batch_idx + 1}/{num_batches}")
            
            batch_indices = []
            
            # Select an anchor and positive pair
            anchor_class = random.choice(self.valid_class[:10])  # Randomly select a class from the first 10 classes
            print(f"Selected anchor class: {anchor_class}")
            self.valid_class.remove(anchor_class)  # Remove the selected class from the valid class list
            self.valid_class.append(anchor_class)  # Add the selected class to the end of the list
            
            # Select least used images for anchor and positive
            selected_anchor_indices = self.least_used_items(2, self.valid_pair[anchor_class])
            batch_indices.extend(selected_anchor_indices)  # Add anchor and positive indices to the batch

            # Select a negative class
            negative_class = None
            temp_heap = []
            
            while self.class_heap:
                count, person = heapq.heappop(self.class_heap)  # Select the least used class
                if person != anchor_class:
                    if random.random() < 0.3:  # Adds a 30% chance to skip this class
                        temp_heap.append((count, person))  # Keep track of popped items to restore later
                        continue
                    negative_class = person
                    break
                else:
                    temp_heap.append((count, person))  # Keep track of popped items
            
            # Restore popped items into the heap
            for item in temp_heap:
                heapq.heappush(self.class_heap, item)
            
            if not negative_class:
                # Fallback to random selection if no negative class was found
                negative_class = random.choice([c for c in self.all_classes if c != anchor_class])
            
            print(f"Selected negative class: {negative_class}")
            self.class_count[negative_class] += 1
            heapq.heappush(self.class_heap, (self.class_count[negative_class], negative_class))
            
            # Select least used images for the negative class
            selected_negative_indices = self.least_used_items(1, self.person_to_indices[negative_class])
            batch_indices.extend(selected_negative_indices)  # Add negative indices to the batch

            # Fill remaining indices to meet the batch size
            remaining_slots = self.batch_size - len(batch_indices)
            if remaining_slots > 0:
                remaining_indices = [idx for idx in range(len(self.dataset.id_image_pair)) if idx not in batch_indices]
                if remaining_indices:
                    remaining_indices = self.least_used_items(remaining_slots, remaining_indices)
                    batch_indices.extend(remaining_indices)  # Add remaining indices to the batch
            
            # Update the image count for the batch
            for idx in batch_indices:
                self.image_count[idx] += 1
            
            # Print the batch indices for debugging
            print(f"Batch indices: {batch_indices}")
            if len(batch_indices) != len(set(batch_indices)):
                print(f"WARNING: Found duplicate indices in batch!")
                print(f"Original length: {len(batch_indices)}, Unique length: {len(set(batch_indices))}")
                print(f"Number of duplicates: {len(batch_indices) - len(set(batch_indices))}")
            else:
                print("No duplicate indices found in this batch.")
            yield batch_indices  # Yield the batch indices ensuring batch size is maintained
    
    def __len__(self):
        # Returns the number of batches in the dataset
        return (len(self.dataset.id_image_pair) + self.batch_size - 1) // self.batch_size
    

root_dir = 'Data/lfw-deepfunneled/lfw-deepfunneled/'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = LFWDataset(root_dir, split = 'train', transform = transform, mode = 'hard')
sampler = BalancedHardTripletSamplerTest(dataset, batch_size=300, min_samples_per_class=2)
dataloader = DataLoader(dataset, batch_sampler=sampler)
# Check the number of batches
print(f"Number of batches: {len(dataloader)}")
# loop through the dataloader and print the batch indices
for batch_idx, (_, person_ids) in enumerate(dataloader):
    print(f"Person IDs: {person_ids}")
    # Simple duplicate check
    unique_count = len(torch.unique(person_ids))
    total_count = len(person_ids)
    print(f"Unique person IDs: {unique_count} out of {total_count}")
    
    if unique_count < total_count:
        print(f"Found {total_count - unique_count} repeated person IDs")
    else:
        print("No repeated person IDs")
        
    if batch_idx == 1:  # Limit to first 2 batches for brevity
        break
