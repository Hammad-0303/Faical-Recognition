
import random
import heapq
from torch.utils.data import Sampler
from collections import defaultdict

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

