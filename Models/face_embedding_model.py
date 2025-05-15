import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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