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
        total_people = len(all_people) #Total number of people in the dataset
        train_people = all_people[:int(0.8*total_people)] #80% of people in training set
        valid_people = all_people[int(0.8*total_people):int(0.9*total_people)] #10% of people in validation set
        test_people = all_people[int(0.9*total_people):] #10% of people in test set

        if split == 'train':
            selected_people = train_people
        elif split == 'val':
            selected_people = valid_people
        elif split == 'test':
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
        return min(150000, max(1000,3*(self.num_pairs))) #Ensuring that the number of triplets is at least 1000
    
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
   


    





            







        
        
