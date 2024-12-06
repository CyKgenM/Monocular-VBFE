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
from torchvision.models import vgg16, VGG16_Weights
from pointnet2_ssg import get_model as PNet2


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class SpatialBlock(torch.nn.Module):
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

        pointnet = PNet2(40, normal_channel=False)
        #checkpoint = torch.load("pnet2_weights.pth")
        checkpoint = torch.load("pnet2_weights_ssg.pth")
        pointnet.load_state_dict(checkpoint['model_state_dict'], strict=False)
        pointnet.eval()
        pointnet.to(device)
        self.pc_feature_extractor = pointnet

    def forward(self, image, pointcloud):
        image = self.rgb_feature_extractor(image)
        pointcloud = self.pc_feature_extractor(pointcloud)[0]
        cat = torch.cat((image, pointcloud), dim=-1)

        return cat.squeeze(0)

