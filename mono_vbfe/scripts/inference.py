#!/usr/bin/env python

import torch
import rospy
from pytorch_tcn import TCN
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from include.spatial import SpatialBlock as sb
import time
import cv2
from main import mono
from std_msgs.msg import Float32  # Import ROS message type

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
force_pub = rospy.Publisher("/dvrk/force_feedback", Float32, queue_size=10)

# Load the TCN model
model = TCN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device(device)))
model.eval()
model.reset_buffers()

# Load Metric3D model
metric3d = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True)
intrinsics = [833.170, 909.161, 275.833, 297.017]

# GStreamer pipeline for ECM camera
GSTREAMER_PIPELINE = "gst-launch-1.0 decklinkvideosrc ! videoconvert ! appsink"

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

def capture_image_from_gstreamer():
    cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        rospy.logerr("Error: Unable to open GStreamer pipeline.")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        rospy.logerr("Error: Unable to capture image from GStreamer.")
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def capture_data():
    image = capture_image_from_gstreamer()
    if image is None:
        rospy.logwarn("Image capture failed. Skipping frame.")
        return None, None
    depth = mono(image, metric3d, intrinsics)
    return image, depth_to_pointcloud(depth, intrinsics)

def send_force_feedback(force):
    # Publish the force feedback to the ROS topic
    msg = Float32()
    msg.data = force
    force_pub.publish(msg)
    rospy.loginfo(f"Published Force Feedback: {force}")

# Main Loop
if __name__ == "__main__":
    sb = sb()
    block_size = 5
    sequence_length = 15
    rate = rospy.Rate(2)  # Adjust sampling rate (e.g., 2 Hz)
    start_time = time.time()
    try:
        while not rospy.is_shutdown():
            image, pointcloud = capture_data()
            if image is None or pointcloud is None:
                continue
            processed_image = preprocess_image(image)
            processed_pointcloud = torch.tensor(pointcloud.transpose(0, 1), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                features = sb(processed_image, processed_pointcloud).transpose(1, -1)
                blocks = [features[:, :, i : i + block_size] for i in range(0, sequence_length, block_size)]
                outputs = [F.softplus(model(block, inference=True)[:, :, -1]) for block in blocks]
                force_prediction = torch.cat(outputs, dim=2)
            for o in outputs:
                send_force_feedback(o.item())
                actual_time = time.time() - start_time
                t = Float32()
                t.data = actual_time
                force_pub.publish(t)
                rospy.loginfo(f"Time spent doing so: {actual_time} s")
                rate.sleep()  # Maintain consistent publishing rate
    except rospy.ROSInterruptException:
        rospy.loginfo("Force feedback node shutting down.")
