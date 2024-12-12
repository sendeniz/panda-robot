import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from utils.dataloader import ImageWaypointDataset

#Hyperparameters
latent_dim = 256
lr = 1e-3
batch_size = 128
epochs = 200

# Define the transformations
img_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256 (you can change the size)
    transforms.ToTensor(),          # Convert image to tensor
])

waypoint_transform = transforms.Compose([
    transforms.ToTensor(),          # Convert image to tensor
])

# Create the dataset and dataloader
dataset = ImageWaypointDataset(img_dir='data/bridge/img', waypoint_dir='data/bridge/paths', img_transform=img_transform, waypoint_transform=waypoint_transform)
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Define the size of train and test sets
train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing

# Split the dataset into train and test sets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoader for both train and test datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

