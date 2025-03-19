import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepLabv3p(nn.Module):
    def __init__(self, backbone, classifier):
        super(DeepLabv3p, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        prediction = self.classifier(features)
        upsampled_prediction = F.interpolate(prediction, size=input_shape, mode='bilinear', align_corners=False)
        return {'out': upsampled_prediction}

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.encoder = ASPP(in_channels, aspp_dilate)

        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, features):
        encoder_output = self.encoder(features['out'])
        decoder_output = self.decode(encoder_output, features['low_level'])
        return decoder_output

    def decode(self, encoder_output, low_level_features):
        low_level_features = self.project(low_level_features)
        upsampled_encoder_output = F.interpolate(encoder_output, size=low_level_features.shape[2:], mode='bilinear', align_corners=False)
        concat_output = torch.cat( [ low_level_features, upsampled_encoder_output ], dim=1 )
        prediction = self.classifier(concat_output)
        return prediction

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        modules = [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPPooling, self).__init__(*modules)

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)                     # 32x256x1x1
        out = F.interpolate(x, size=size, mode='bilinear', align_corners=False) # 32x256x17x17
        return out

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

if __name__ == '__main__':
    print('hello')
    in_planes = 2048
    low_level_planes = 256
    num_classes = 21
    aspp_dilate = [12, 24, 36]    # paper: [6, 12, 18]

    DeepLabHeadV3Plus(in_planes, low_level_planes, num_classes, aspp_dilate)
