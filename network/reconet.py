import torch
import torch.nn as nn
from torchvision import models

from .deeplabv3 import DeepLabv3p

class ReCoNet(DeepLabv3p):
    def __init__(self, backbone, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super().__init__(backbone, in_channels, low_level_channels, num_classes, aspp_dilate)
        self.representation = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, low_level_channels, 1)
        )
    def forward(self, x):
        classifier_output = super().forward(x)
        prediction, decoder_output = classifier_output['out'], classifier_output['decoder']
        representation = self.representation(decoder_output)
        representation = F.interpolate(representation, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return {
            'out': prediction,
            'reco': representation,
        }

if __name__ == '__main__':
    from torchvision import models
    backbone = models._utils.IntermediateLayerGetter(
        models.resnet101(pretrained=True),
        {'layer4': 'out', 'layer1': 'low_level'}
    )
    model = ReCoNet(backbone, 2048, 256, 21, [12,24,36])
