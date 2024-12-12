import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np 
from utils import waypoints_from_bridge_build
# Define a custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB") #.convert("L")  # Convert to grayscale if needed
        if self.transform:
            image = self.transform(image)
            #print("shape img", image.shape)

        return image


class ImageWaypointDataset(Dataset):
    def __init__(self, img_dir, waypoint_dir, img_transform=None, time_step = -1):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.img_dir = img_dir
        self.waypoint_dir = waypoint_dir
        self.img_transform = img_transform
        self.image_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        self.waypoint_files = [f for f in os.listdir(waypoint_dir) if os.path.isfile(os.path.join(waypoint_dir, f))]
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        waypoint_path = os.path.join(self.waypoint_dir, self.waypoint_files[idx])
        image = Image.open(img_path).convert("RGB") #.convert("L")  # Convert to grayscale if needed
        waypoint = np.load(waypoint_path)
        pos, quat = waypoints_from_bridge_build(waypoint)
        if self.img_transform:
            image = self.img_transform(image)
        return image, pos


def test():

    # Define the transformations
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256 (you can change the size)
        transforms.ToTensor(),          # Convert image to tensor
    ])

    waypoint_transform = transforms.Compose([
        transforms.ToTensor(),          # Convert image to tensor
    ])


    dataset = ImageWaypointDataset(img_dir='data/bridge/img', waypoint_dir='data/bridge/paths', 
                                    img_transform=img_transform, 
                                    waypoint_transform=waypoint_transform)
    
    loader = DataLoader(dataset = dataset, batch_size = 32, shuffle = False)
    
    for idx, (x, y) in enumerate(loader):
        #print("idx:", idx)
        #print("image x:", x.shape)
        print("waypoint y:", y.shape)
        pass

test()