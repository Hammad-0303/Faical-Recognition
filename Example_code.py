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
    def __init__(self, root_dir, split="train", transform=None):
        """
        LFW Dataset with dynamic triplet sampling.
        Args:
            root_dir: Root directory of LFW dataset
            split: "train", "val", or "test"
            transform: Optional transform to be applied on images
        """
        # Set random seed for reproducibility
        random.seed(42)  # Ensures consistent dataset splitting across runs
        torch.manual_seed(42)  # Ensures deterministic PyTorch operations
        
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Store all person identities (including those with only 1 image)
        self.all_identities = {}  # Stores {person: [image paths]} for ALL identities
        
        # Specifically store identities with 2+ images (for anchor-positive pairs)
        self.valid_identities = {}  # Stores {person: [image paths]} for identities with 2+ images
        
        # Store all images across all identities in this split
        self.all_images = []
        
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

        # Step 4: Store ALL selected identities and their images
        for person in selected_people:
            images = person_images[person]
            # Store all identities and their images
            self.all_identities[person] = images
            self.all_images.extend(images)
            
            # Also store valid identities (those with 2+ images)
            if len(images) > 1:
                self.valid_identities[person] = images

        # Calculate the number of valid anchor-positive pairs
        self.num_pairs = sum(len(imgs) * (len(imgs) - 1) // 2 for imgs in self.valid_identities.values())
        
        # Clear memory
        del person_images
        del all_people
        del train_people
        del val_people
        del test_people
        gc.collect()
        
        print(f"Split: {split}, Total identities: {len(self.all_identities)}, Valid identities: {len(self.valid_identities)}")
        print(f"Total images: {len(self.all_images)}, Possible pairs: {self.num_pairs}")

    def __len__(self):
        """
        Returns a length proportional to the number of possible anchor-positive pairs.
        This ensures good coverage of the dataset during training.
        """
        # Multiply by a factor to ensure good coverage (adjustable)
        return max(1000, self.num_pairs * 3)  # Ensure at least 1000 examples

    def __getitem__(self, index):
        """
        Dynamically generates triplets using epoch-based sampling.
        """
        # Randomly select an anchor identity with at least 2 images
        anchor_person = random.choice(list(self.valid_identities.keys()))
        
        # Get images for this person
        anchor_images = self.valid_identities[anchor_person]
        
        # Randomly select anchor and positive (different images of same person)
        anchor_path, positive_path = random.sample(anchor_images, 2)
        
        # Select a random negative from a different identity
        # Note: We're selecting from ALL identities, not just valid ones
        negative_candidates = [p for p in self.all_identities.keys() if p != anchor_person]
        negative_person = random.choice(negative_candidates)
        negative_path = random.choice(self.all_identities[negative_person])
        
        # Load images
        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

