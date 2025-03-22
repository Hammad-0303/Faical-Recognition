#%%
# Different code samples for creating a dataloader for triplet loss
# Makes data loader dataset which splits into train, test and val and also creates triplets of size (batch size, height, width, channels)
# Creates unique sampling of triplets, occupies alot of memory and no inherent advantage
import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from itertools import combinations
import gc

class LFWDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Efficiently stores only triplet indices while ensuring uniqueness.
        """
        # Set random seed for reproducibility
        random.seed(42)  # Ensures consistent dataset splitting across runs
        torch.manual_seed(42)  # Ensures deterministic PyTorch operations
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.identity_dict = {}  # Stores {person: [image paths]}
        self.all_images = []  # All images in this split
        self.image_to_idx = {}  # Maps image path -> index
        self.triplet_indices = []  # Stores (anchor_idx, positive_idx, negative_idx)

        all_people = []
        person_images = {}

        # Step 1: Collect all identities and images
        for person in os.listdir(root_dir):
            person_dir = os.path.join(root_dir, person)
            if os.path.isdir(person_dir):
                images = [os.path.join(person_dir, img) for img in os.listdir(person_dir) if img.endswith('.jpg')]
                
                if images:
                    person_images[person] = images
                    all_people.append(person)

        # Step 2: Shuffle people randomly (for a fair split)
        random.shuffle(all_people)
        total_images = sum(len(imgs) for imgs in person_images.values())

        train_people, val_people, test_people = [], [], []
        train_count, val_count, test_count = 0, 0, 0

        for person in all_people:
            images = person_images[person]
            num_images = len(images)

            if train_count / total_images < 0.8:
                train_people.append(person)
                train_count += num_images
            elif val_count / total_images < 0.1:
                val_people.append(person)
                val_count += num_images
            else:
                test_people.append(person)
                test_count += num_images

        # Step 3: Assign selected people based on split
        selected_people = train_people if split == "train" else val_people if split == "val" else test_people

        # Step 4: Store image lists & create index mapping
        idx = 0
        for person in selected_people:
            images = person_images[person]
            if len(images) > 1:
                self.identity_dict[person] = images  # Store only if at least 2 images exist

            for img_path in images:
                self.image_to_idx[img_path] = idx  # Assign an index to each image
                self.all_images.append(img_path)
                idx += 1

        # Step 5: Precompute **unique** triplet indices (anchor, positive, negative)
        self.precompute_triplet_indices()
        random.shuffle(self.triplet_indices)  # Shuffle for random sampling
        del person_images  # Clear memory
        del all_people  # Clear memory
        del train_people  # Clear memory
        del val_people  # Clear memory
        del test_people  # Clear memory
        del selected_people  # Clear memory
        del self.image_to_idx  # Clear memory
        gc.collect()  # Clear memory

    def precompute_triplet_indices(self):
        """Generates all possible unique triplet indices and stores only indices (not full paths)."""
        for person, images in self.identity_dict.items():
            if len(images) < 2:
                continue  # Ignore identities with fewer than 2 images

            # Generate all unique (Anchor, Positive) pairs
            anchor_positive_pairs = list(combinations(images, 2))

            # Select negative samples (ensuring they are from another person)
            for anchor_path, positive_path in anchor_positive_pairs:
                for negative_path in self.all_images:
                    if person not in negative_path:  # Ensure negative is from another identity
                        anchor_idx = self.image_to_idx[anchor_path]
                        positive_idx = self.image_to_idx[positive_path]
                        negative_idx = self.image_to_idx[negative_path]
                        self.triplet_indices.append((anchor_idx, positive_idx, negative_idx))

    def __len__(self):
        return len(self.triplet_indices)  # Number of unique triplets stored as indices

    def __getitem__(self, index):
        """
        Loads images dynamically based on precomputed triplet indices.
        """
        anchor_idx, positive_idx, negative_idx = self.triplet_indices[index]

        # Get image paths from indices
        anchor_path = self.all_images[anchor_idx]
        positive_path = self.all_images[positive_idx]
        negative_path = self.all_images[negative_idx]

        # Load images & apply transformations
        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative
    

#%%
#Same dataloader but uses less memory as it is not storing all the images in memory
import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import gc

class LFWDataset(Dataset):
    def __init__(self, root_dir, split = 'train', transform = None):
        #Splits the LFW dataset into train, validation, and test sets
        #Generates triplets (Anchor, Positive, Negative) dynamically
        #Parameters:
        #root_dir: Path to LFW dataset
        #split: 'train', 'val', or 'test'
        #transform: Image transformation
        random.seed(42) #For reproducibility
        torch.manual_seed(42) #For reproducibility
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.all_images = []  #Stores all images in the dataset
        self.all_identities = {}   #Stores all identities of form {identity: [image1, image2, ...]} even if there is 1 image per person
        self.valid_identities = {}  #Stores all identities with more than 1 image of form {identity: [image1, image2, ...]}
        all_people = []
        person_images = {}      
        for person in os.listdir(root_dir):
            person_dir = os.path.join(root_dir, person)
            if os.path.isdir(person_dir):
                images = [os.path.join(person_dir, img) for img in os.listdir(person_dir) if img.endswith('.jpg')]
                if images: #If there are images in the directory
                    all_people.append(person)
                    person_images[person] = images


        random.shuffle(all_people)
        total_images = sum(len(imgs) for imgs in person_images.values())    
        train_people, test_people, valid_people = [], [], []
        train_count, test_count, valid_count = 0, 0, 0
        for person in all_people:
            if (train_count/total_images) < 0.8:
                train_people.append(person)
                train_count += len(person_images[person])
                continue
            elif (valid_count/total_images) < 0.9:
                valid_people.append(person)
                valid_count += len(person_images[person])
                continue
            else:
                test_people.append(person)
                test_count += len(person_images[person])
                continue
        
        if split == 'train':
            selected_people = train_people
        elif split == 'val':
            selected_people = valid_people
        else:
            selected_people = test_people
        
        for person in selected_people:
            images = person_images[person]
            self.all_images.extend(images)
            self.all_identities[person] = images
            if len(images) > 1:
                self.valid_identities[person] = images
        
        #Calculting the number of anchor postitive pairs
        self.num_pairs = sum((len(images)*(len(images)-1))//2 for images in self.valid_identities.values())

        #Deleting unused variables
        del all_people, person_images, train_people, test_people, valid_people
        gc.collect()
    
    def __len__(self):
        return max(1000,3*(self.num_pairs)) #Ensuring that the number of triplets is at least 1000
    
    def __getitem__(self, idx):
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

