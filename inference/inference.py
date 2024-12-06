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
import rospy
from libs import sb
from pytorch_tcn import TCN
from torchvision.transforms import functional as TF
import torch.nn.functional as F
import time
import cv2
import serial
import matplotlib.pyplot as plt
import numpy as np
import gi
from std_msgs.msg import Float32

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
rospy.init_node("force_feedback_node", anonymous=True)

# Publisher for Force Feedback
force_pub = rospy.Publisher("/console/force_feedback", Float32, queue_size=10)

# Load the TCN model
model = TCN(4608, [64, 128, 256], kernel_size=3, dropout=0.3, use_norm='batch_norm', output_projection=1).to(device)
model_path = "/home/ros2/Martin/inference/model.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model.eval()
model.reset_buffers()

# Load Metric3D model
metric3d = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True)
intrinsics = [833.170, 909.161, 275.833, 297.017]

pnet_path = "/home/ros2/Martin/inference/pnet2_weights_ssg.pth"

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
    rgb = rgb[None, :, :, :].cuda()
    
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
    return TF.to_tensor(TF.resize(image, (224, 224))).unsqueeze(0)

def depth_to_pointcloud(depth, intrinsics):
    fx, fy, cx, cy = intrinsics
    H, W = depth.shape
    u, v = torch.meshgrid(torch.arange(0, W), torch.arange(0, H), indexing="xy")
    u, v = u.flatten(), v.flatten()
    z = depth.flatten()
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return torch.stack((x, y, z), dim=1)

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


def send_force_feedback(force):
    # Publish the force feedback to the ROS topic
    if force > 0.05:
        msg = Float32()
        msg.data = force
        force_pub.publish(msg)
        rospy.loginfo(f"Published Force Feedback: {force}")


# Main program setup
if __name__ == "__main__":
    sb = sb()
    block_size = 5
    sequence_length = 15
    rate = rospy.Rate(6) # 6 Hz
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
        while running or not rospy.is_shutdown():
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
            buff = []
            for k in range(sequence_length):
                image, pointcloud = capture_data()
                if image is None or pointcloud is None:
                    continue
                processed_image = preprocess_image(image)
                processed_pointcloud = torch.tensor(pointcloud.transpose(0, 1), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    features = sb(processed_image, processed_pointcloud)
                    buff.append(features)

            buff = torch.stack(buff, dim=-1)
            blocks = [buff[:, :, i : i + block_size] for i in range(0, sequence_length, block_size)]
            with torch.no_grad():
                outputs = []
                measurements = []
                for block in blocks:
                    output = abs(model(block, inference=True)[:, :, -1])
                    measurement = ser.readline().decode().strip()
                    outputs.append(output)
                    measurements.append(measurement)
            
            for o in outputs:
                for m in measurements:
                    pred = o.item()
                    
                    measurement = float(ser.readline().decode().strip())
                    if m:
                        send_force_feedback(pred)
                        actual_time = time.time() - start_time
                        t = Float32()
                        t.data = actual_time
                        force_pub.publish(t)
                        rospy.loginfo(f"Time spent doing so: {actual_time} s")
                        abs_err = abs(m - o.item())
                        errors.append(abs_err)
                        delays.append(actual_time)
                        print(f"Validation Error: {abs_err} N \nTime spent doing so: {actual_time} s")
                    rate.sleep(0.5)
    except rospy.ROSInterruptException or Exception as e:
        running = False
        if e:
            print(e)
        rospy.loginfo("Force feedback node shutting down.")
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