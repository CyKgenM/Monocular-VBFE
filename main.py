'''
Metric3D:

BSD 2-Clause License

Copyright (c) 2024, Wei Yin and Mu Hu

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import torch
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode
import torch.nn.functional as F
from include import sb
from pytorch_tcn import TCN


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def depth_to_pointcloud(depth, intrinsics):
    # Intrinsics parameters
    fx, fy, cx, cy = intrinsics
    H, W = depth.shape

    # Generate pixel grid
    u, v = torch.meshgrid(
        torch.arange(0, W, device=depth.device),
        torch.arange(0, H, device=depth.device),
        indexing='xy'
    )
    u, v = u.flatten(), v.flatten()  # Flatten to vector for easier handling

    # Convert depth image to 3D points
    z = depth.flatten()  # Depth values
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack to form [N, 3] point cloud
    points = torch.stack((x, y, z), dim=1)
    
    return points


def extract_image_number(filename):
    match = re.search(r'frame_(\d+)\.png', filename)
    return int(match.group(1)) if match else -1  # Extract number, fallback -1 for safety

def mono(rgb_file, model, intrinsic):
    #### prepare data
    #intrinsic = [212.010, 212.010, 213.846, 121.795] #[707.0493, 707.0493, 604.0814, 180.5066]
    rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]
    
    # Adjust input size to fit model requirements
    input_size = (616, 1064)  # For ViT model; use (544, 1216) for ConvNeXt model
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    
    # Scale intrinsic parameters
    intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
    
    # Pad the image to match model input size
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    
    # Normalize the image
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()
    
    # Load pre-trained model and perform inference
    #model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True) # was 'metric3d_vit_small'
    model.cuda().eval()
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({'input': rgb})
    
    # Remove padding
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad_info[0]: pred_depth.shape[0] - pad_info[1], pad_info[2]: pred_depth.shape[1] - pad_info[3]]
    
    # Upsample to original size
    #pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], size=(224, 224), mode='bilinear').squeeze()
    #print(pred_depth.size())
    # Convert depth to metric space (if needed)
    canonical_to_real_scale = intrinsic[0] / 1000.0  # Adjust based on focal length of canonical camera
    pred_depth = pred_depth * canonical_to_real_scale
    pred_depth = torch.clamp(pred_depth, 0, 300)  # Clamping depth values for visualization
    #print(pred_depth.size())

    return pred_depth

class RPC_Dataset(Dataset):
    def __init__(self, image_dir, forces, image_transform=None, pointcloud_transform=None):
        self.image_dir = image_dir
        self.labels = pd.read_csv(forces, header=None)
        
        ### Intrinsic parameters: [fx, fy, cx, cy]
        # For ECM left: [833.170, 909.161, 275.833, 297.017]
        # For ECM right: [832.659, 908.230, 407.354, 304.965]
        # For Realsense: [212.010, 212.010, 213.846, 121.795]
        self.intrinsics = [833.170, 909.161, 275.833, 297.017] 
        self.metric3d = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True) # was 'metric3d_vit_large'

        self.sb = sb()
        
        rgb_transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        self.image_transform = rgb_transform
        
        self.seq_length = 15

        # Get a list of image and point cloud file names (assuming they match)
        self.image_filenames = sorted(os.listdir(image_dir), key=extract_image_number)
        

    def __len__(self):
        #print(len(self.image_filenames))
        return int(len(self.image_filenames) / self.seq_length)

    def __getitem__(self, idx):
        cats = []

        # Loop over the sequence length to get 15 frames
        for i in range(self.seq_length):
            # Load the image
            img_path = os.path.join(self.image_dir, self.image_filenames[idx * self.seq_length + i])
            image = read_image(img_path, ImageReadMode.RGB)
            
            # Predict depth map
            pred_depth = mono(img_path, self.metric3d, self.intrinsics)
            
            # Apply base transformations
            image = self.image_transform(image)

            # Convert depth map to point cloud tensor
            pointcloud_tensor = depth_to_pointcloud(pred_depth, self.intrinsics)  # Shape (N, 3) where N is the number of points
            
            
            pointcloud_tensor = pointcloud_tensor.transpose(0, 1)
            image, pointcloud_tensor = image.to(device), pointcloud_tensor.to(device)
            
            with torch.no_grad():
                cat_features = self.sb(image.unsqueeze(0), pointcloud_tensor.unsqueeze(0))
    
            cats.append(cat_features)
        
        
        cats = torch.stack(cats, dim=0)
        # Retrieve the label for this sequence
        label = self.labels.iloc[idx * self.seq_length + self.seq_length - 1, 2]
        
        return cats, torch.tensor(label, dtype=torch.float32)


# Initialize the dataset

train_image_dir = 'frame/ECM_frames/left'
train_forces = 'frame/ECM_frames/labels.csv'

test_image_dir = 'frame/ECM_frames/left_test'
test_forces = 'frame/ECM_frames/test.csv'
'''
# Debugging dataset
train_image_dir = 'frame/ECM_frames/right'
train_forces = 'frame/ECM_frames/debug.csv'

test_image_dir = 'frame/ECM_frames/right_test'
test_forces = 'frame/ECM_frames/debug_test.csv'
'''
train_dataset = RPC_Dataset(train_image_dir, train_forces)
test_dataset = RPC_Dataset(test_image_dir, test_forces)
#augmented_test_dataset = RPC_Dataset(test_image_dir, test_pointcloud_dir, test_forces, image_transform=rgb_augment, pointcloud_transform=pc_augment)
#augmented_test_dataset = ConcatDataset([test_dataset, augmented_test_dataset])

'''
data = dataset.__getitem__(5)
image, pointcloud = data[0], data[1]
visualize_rgb(image)
visualize_point_cloud(pointcloud)
'''
# Use DataLoader to batch
train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False)

'''
class RPC_TCN(nn.Module):
    def __init__(self, input_size=4608, output_size=1, num_channels=[64, 128, 256], kernel_size=3, dropout=0.3):
        super(RPC_TCN, self).__init__()

        self.tcn = TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout, use_norm='batch_norm', output_projection=1) # old use_norm='layer_norm'
        
    def forward(self, cat_input):   
        # TB forward prop 
        #cat_input = cat_input.unsqueeze(0) # Only when batch_size = 1
        cat_input = cat_input.transpose(1, -1)
        #print(cat_input.size())
        
        force = self.tcn(cat_input)  # Apply TCN layer
        #print(force.size())
        force = F.softplus(force[:, :, -1])
        
        return force

'''
model = TCN(4608, [64, 128, 256], kernel_size=3, dropout=0.3, use_norm='batch_norm', output_projection=1).to(device)

#model = RPC_TCN().to(device)
#print(model)

# Mean Squared Error
loss_fn = nn.MSELoss()
#loss_fn = nn.SmoothL1Loss() # MAE used instead of MSE on critical losses to prevent gradient explosion
#loss_fn = nn.L1Loss()
#optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1)



def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (cats, f) in enumerate(dataloader):
        #torch.cuda.empty_cache()
        cats, f = cats.float(), f.float()
        cats, f, = cats.to(device), f.to(device)
        #print(f)
        
        # Compute prediction error
        cats = cats.transpose(1, -1)
        pred = abs(model(cats)[:, :, -1])
        pred = pred.squeeze(0)
        pred = pred.squeeze(-1)
        #print(pred.size())
        loss = loss_fn(pred, f)
        
        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        if batch % 25 == 0:
            loss, current = loss.item(), (batch + 1) * len(cats)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    total_absolute_error, test_loss = 0, 0
    
    with torch.no_grad():
        for cats, f in dataloader:
            cats, f = cats.float(), f.float()
            cats, f, = cats.to(device), f.to(device)
            
            # Forward pass: Compute predictions
            cats = cats.transpose(1, -1)
            pred = abs(model(cats)[:, :, -1])
            pred = pred.squeeze(0)  # Ensure correct shape if necessary
            pred = pred.squeeze(-1)
            #print(pred.size())
            
            # Accumulate Mean Squared Error loss (MSE) as used in training
            test_loss += loss_fn(pred, f).item()
            
            # Calculate Mean Absolute Error (MAE) for this batch
            total_absolute_error += torch.sum(torch.abs(pred - f)).item()


    # Calculate average losses
    avg_mse_loss = test_loss / num_batches
    mae = total_absolute_error / size



    print(f"Test Results: \n MAE: {mae:.5f}, Avg MSE Loss: {avg_mse_loss:.5f} \n")

    return mae, avg_mse_loss

epochs = 50
abs_err = []
avg_err = []
actual_ep = 0
try:
    for t in range(epochs):
        torch.cuda.empty_cache()
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        mae, mse = test(test_dataloader, model, loss_fn)
        avg_err.append(mse)
        abs_err.append(mae)
        scheduler.step()
        actual_ep += 1
    print("Done!")
except KeyboardInterrupt():
    pass
finally:
    torch.save(model.state_dict(), "model.pth")
    torch.cuda.empty_cache()
    print("Saved PyTorch Model State to model.pth")
    
    epoch_ls = list(range(1, actual_ep + 1))
    
    # Plot MAE and MSE
    plt.figure(figsize=(10, 6))  # Set the figure size
    
    # Plot MAE
    plt.plot(epoch_ls, abs_err, label='Mean Absolute Error (MAE)', color='blue', marker='o')
    
    # Plot MSE
    plt.plot(epoch_ls, avg_err, label='Mean Squared Error (MSE)', color='green', marker='x')
    
    # Add titles and labels
    plt.title('Training Errors Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    
    # Add a legend
    plt.legend(fontsize=12)
    
    # Show grid for better readability
    plt.grid(True)
    
    # Display the plot
    #plt.show()
    plt.savefig('training_errors.png', dpi=300)