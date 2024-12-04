#!/usr/bin/env python

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
'''
PointNet implementation:

MIT License

Copyright (c) 2019 benny

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import torch
import torch.nn as nn
#import rospy
import signal
from pytorch_tcn import TCN
from torchvision.transforms import functional as TF
from torchvision.models import vgg16, VGG16_Weights
import torch.nn.functional as F
import time
import cv2
import serial
import matplotlib.pyplot as plt
import numpy as np
import gi
#from std_msgs.msg import Float32

gi.require_version('Gst', '1.0')
from gi.repository import Gst


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# ROS Node Initialization
#rospy.init_node("force_feedback_node", anonymous=True, disable_signals=True)

# Publisher for Force Feedback
#force_pub = rospy.Publisher("/console/force_feedback", Float32, queue_size=10)

# Load the TCN model
model = TCN(4608, [64, 128, 256], kernel_size=3, dropout=0.3, use_norm='batch_norm', output_projection=1).to(device)
model_path = "/home/ros2/Martin/inf/model.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model.eval()
model.reset_buffers()

# Load Metric3D model
metric3d = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True)
intrinsics = [833.170, 909.161, 275.833, 297.017]

pnet_path = "/home/ros2/Martin/inf/pnet2_weights_ssg.pth"

port = '/dev/ttyUSB0'
ser = serial.Serial(port, baudrate=115200)

Gst.init(None)

# GStreamer pipeline for ECM camera
GSTREAMER_PIPELINE = """
        decklinkvideosrc mode=pal device-number=0 ! 
        videorate ! video/x-raw,framerate=30/1 ! 
        videoconvert ! video/x-raw,format=RGB ! appsink name=appsink
    """

pipeline = Gst.parse_launch(GSTREAMER_PIPELINE)
# Access the appsink element
appsink = pipeline.get_by_name("appsink")
appsink.set_property("emit-signals", True)
appsink.set_property("sync", False)

### Dependencies:

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points=None):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNet(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(PointNet, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)
        self.fc = nn.Linear(num_class, 512)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = self.fc(x)
        #x = F.log_softmax(x, -1)


        return x, l3_points

class SpatialBlock(nn.Module):
    def __init__(self):
        super(SpatialBlock, self).__init__()

        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        vgg.eval()
        vgg.to(device)
        self.rgb_feature_extractor = torch.nn.Sequential(
            *list(vgg.features.children()),   # Convolutional layers
            torch.nn.AdaptiveAvgPool2d((7, 7)),     # Ensure 7x7 output size
            torch.nn.Flatten(),                     # Flatten to (batch_size, 25088)
            *list(vgg.classifier.children())[:-1]  # Use up to the second to last FC layer (4096 output)
        )

        pointnet = PointNet(40, normal_channel=False)
        #checkpoint = torch.load("pnet2_weights.pth")
        checkpoint = torch.load(pnet_path)
        pointnet.load_state_dict(checkpoint['model_state_dict'], strict=False)
        pointnet.eval()
        pointnet.to(device)
        self.pc_feature_extractor = pointnet

    def forward(self, image, pointcloud):
        image = self.rgb_feature_extractor(image)
        pointcloud = self.pc_feature_extractor(pointcloud)[0]
        cat = torch.cat((image, pointcloud), dim=-1)
        #print(cat.size())

        return cat


def mono(rgb_file, model, intrinsic):
    #### prepare data
    #intrinsic = [212.010, 212.010, 213.846, 121.795] #[707.0493, 707.0493, 604.0814, 180.5066]
    #rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]
    rgb_origin = rgb_file[:, :, ::-1]
    
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
    rgb = rgb[None, :, :, :].to(device)
    
    # Load pre-trained model and perform inference
    #model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True) # was 'metric3d_vit_small'
    model.eval().to(device)
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

def preprocess_image(image):
    tensor_image = torch.from_numpy(image).permute(2, 0, 1).float().to(device)
    tensor_image = F.interpolate(tensor_image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)

    return tensor_image

def depth_to_pointcloud(depth, intrinsics):
    fx, fy, cx, cy = intrinsics
    H, W = depth.shape
    u, v = torch.meshgrid(torch.arange(0, W), torch.arange(0, H), indexing="xy")
    u, v = u.flatten().to(device), v.flatten().to(device)
    z = depth.flatten().to(device)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return torch.stack((x, y, z), dim=1)
'''
def capture_image_from_gstreamer():
    cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        rospy.logerr("Error: Unable to open GStreamer pipeline.")
        print("Error: Unable to open GStreamer pipeline.")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        rospy.logerr("Error: Unable to capture image from GStreamer.")
        print("Error: Unable to capture image from GStreamer.")
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
'''

running = True
'''
def signal_handler(sig, frame):
    global running
    print('Ctrl+C detected! Stopping the pipeline...')
    running = False

# Register signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)
'''

# Global variable to store the image
current_image = None

# Function to handle new samples from the appsink
def on_new_sample(sink):
    global current_image  # Reference the global variable
    try:
        sample = sink.emit("pull_sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        
        caps = sample.get_caps()
        # Extract the width and height info from the sample's caps
        height = caps.get_structure(0).get_value("height")
        width = caps.get_structure(0).get_value("width")

        # Get the actual data
        buffer = sample.get_buffer()
        # Get read access to the buffer data
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            raise RuntimeError("Could not map buffer data!")

        numpy_frame = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=map_info.data)

        # Clean up the buffer mapping
        buffer.unmap(map_info)
        
        # Save the numpy frame to the global variable
        current_image = numpy_frame
        return Gst.FlowReturn.OK
    
    except Exception as e:
        print(e)
        return Gst.FlowReturn.ERROR

def capture_data():
    global current_image  # Reference the global variable
    if current_image is None:
        print("Image capture failed. Skipping frame.")
        return None, None
    # Process the captured image (you can apply your depth processing here)
    depth = mono(current_image, metric3d, intrinsics)
    return current_image, depth_to_pointcloud(depth, intrinsics)

# Main program setup
if __name__ == "__main__":
    sb = SpatialBlock()
    block_size = 5
    sequence_length = 15
    errors = []
    delays = []
    running = True

    # Connect the signal to the appsink
    appsink.connect("new-sample", on_new_sample)

    # Start the pipeline
    pipeline.set_state(Gst.State.PLAYING)
    time.sleep(1) # Let the pipeline start up

    # Main Loop
    try:
        while running:
            start_time = time.time()

            message = pipeline.get_bus().timed_pop_filtered(
                100*Gst.MSECOND,
                Gst.MessageType.ERROR | Gst.MessageType.EOS
            )
            if message:
                t = message.type
                if t == Gst.MessageType.ERROR:
                    err, debug = message.parse_error()
                    print(f"GStreamer Error: {err}, {debug}")
                    break
                elif t == Gst.MessageType.EOS:
                    print("End of Stream")
                    break
            
            # Capture image from the global variable
            image, pointcloud = capture_data()
            if image is None or pointcloud is None:
                continue
            
            processed_image = preprocess_image(image)
            processed_pointcloud = torch.tensor(pointcloud.transpose(0, 1), dtype=torch.float32).unsqueeze(0)
            #print(processed_image.size())
            #print(processed_pointcloud.size())
            with torch.no_grad():
                features = sb(processed_image, processed_pointcloud).unsqueeze(-1) #.transpose(1, -1)
                blocks = [features[:, :, i : i + block_size] for i in range(0, sequence_length, block_size)]
                for block in blocks:
                    print(block.size())
                outputs = [F.softplus(model(block, inference=True)[:, :, -1]) for block in blocks]
                model.reset_buffers()
                #force_prediction = torch.cat(outputs, dim=2)
            
            for o in outputs:
                actual_time = time.time() - start_time  # Capture time delay in prediction
                measurement = float(ser.readline().decode().strip())
                if measurement > 0.05:
                    abs_err = abs(measurement - o.item())
                    errors.append(abs_err)
                    delays.append(actual_time)
                    print(f"Validation Error: {abs_err} N \nTime spent doing so: {actual_time} s")
                
                time.sleep(0.5)
    except KeyboardInterrupt or Exception as e:
        running = False
        if e:
            print(e)
    finally:
        pipeline.set_state(Gst.State.NULL)
        ser.close()
        if not Exception:
            # Validation Error
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(errors)), errors, label='Absolute Error', color='red', marker='o')
            plt.title('Validation Error', fontsize=16)
            plt.xlabel('Predictions', fontsize=14)
            plt.ylabel('Error', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.show()
            plt.savefig('validation_err.png', dpi=300)

            # Delays
            delay_sum = 0.0
            for delay in delays:
                delay_sum += delay
            avg_delay = delay_sum/len(delays)
            print(f"Average time delay in computation: {avg_delay} s")