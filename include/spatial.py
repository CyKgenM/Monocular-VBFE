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

