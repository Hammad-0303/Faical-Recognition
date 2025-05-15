import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import gc


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