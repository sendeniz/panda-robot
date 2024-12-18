import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from utils.dataloader import ImageWaypointDataset
from model.vw import VisWaypointing
from loss.vw_loss import vis_waypoint_loss
import os 

#Hyperparameters
latent_dim = 256
lr = 1e-3
batch_size = 32
epochs = 200

# Create the cpts/ directory if it doesn't exist
os.makedirs('cpts', exist_ok=True)

model_name = 'vis2way_vae'

# Define the transformations
img_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256 (you can change the size)
    transforms.ToTensor(),          # Convert image to tensor
])

# Create the dataset and dataloader
dataset = ImageWaypointDataset(img_dir='data/bridge/img', waypoint_dir='data/bridge/paths', img_transform=img_transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the size of train and test sets
train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing

# Split the dataset into train and test sets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoader for both train and test datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, optimizer
model = VisWaypointing(latent_dim=latent_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)

scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=
            len(train_loader),
        anneal_strategy="cos",
    )

# Training loop
model.train()
for epoch in range(epochs):
    train_loss = 0
    #idx, (x, y) in enumerate(loader):
    #for data in train_loader:
    for idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        mu, log_var, pred_pos = model(x)
        loss = vis_waypoint_loss(pred_pos, y, mu, log_var)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset):.4f}')

checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}

# Save the checkpoint in the cpts/ folder
checkpoint_path = f'cpts/{model_name}.cpt'
torch.save(checkpoint, checkpoint_path)

print('=> Model saved.')

# Evaluate on test data
model.eval()
test_loss = 0
with torch.no_grad():
    #for data in test_loader:
    for idx, (x, y) in enumerate(test_loader):
        mu, log_var, pred_pos = model(x)
        #print("y shape:", y.shape)
        loss = vis_waypoint_loss(pred_pos, y, mu, log_var)
        test_loss += loss.item()
print(f'Test Loss: {test_loss / len(test_loader.dataset):.4f}')