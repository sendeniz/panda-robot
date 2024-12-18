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
import numpy as np 
import plotly.graph_objs as go

#Hyperparameters
latent_dim = 256
lr = 1e-3
batch_size = 32
epochs = 200


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

# Save the checkpoint in the cpts/ folder
checkpoint_path = f'cpts/{model_name}.cpt'
checkpoint = torch.load(checkpoint_path)

model.load_state_dict(checkpoint['model_state_dict'])

print('=> Model loaded.')

# Evaluate on test data
model.eval()
test_loss = 0
all_preds = []
all_targets = []
with torch.no_grad():
    #for data in test_loader:
    for idx, (x, y) in enumerate(test_loader):
        mu, log_var, pred_pos = model(x)
        loss = vis_waypoint_loss(pred_pos, y, mu, log_var)
        test_loss += loss.item()
        # Store predictions and targets for later visualization
        all_preds.append(pred_pos.cpu().numpy())
        all_targets.append(y.cpu().numpy())
print(f'Test Loss: {test_loss / len(test_loader.dataset):.4f}')
# Convert the list of predictions and targets to numpy arrays
all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)


# Select a specific batch to visualize (you can loop over all batches here)
#for batch_index in range(all_preds.shape[0]):
for batch_index in range(1):
    # Extract the individual sample's predictions and targets
    pred_sample = all_preds[batch_index]
    target_sample = all_targets[batch_index]

    # Create traces for predictions and targets
    prediction_trace = go.Scatter3d(
        x=pred_sample[:, 0],  # x-coordinates of the waypoints in this sample
        y=pred_sample[:, 1],  # y-coordinates of the waypoints in this sample
        z=pred_sample[:, 2],  # z-coordinates of the waypoints in this sample
        mode='markers+lines',  # Display both points and lines connecting them
        marker=dict(size=5, color='red'),  # Red markers for predictions
        name='Predictions'
    )

    target_trace = go.Scatter3d(
        x=target_sample[:, 0],  # x-coordinates of the target waypoints in this sample
        y=target_sample[:, 1],  # y-coordinates of the target waypoints in this sample
        z=target_sample[:, 2],  # z-coordinates of the target waypoints in this sample
        mode='markers+lines',  # Display both points and lines connecting them
        marker=dict(size=5, color='blue'),  # Blue markers for targets
        name='Targets'
    )

    # Create the layout for the 3D plot
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        title=f"3D Waypoints Visualization for Sample {batch_index + 1}",
    )

    # Create the figure and add the traces
    fig = go.Figure(data=[prediction_trace, target_trace], layout=layout)

    # Show the interactive 3D plot
    fig.show()