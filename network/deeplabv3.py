import torch.nn as nn
from torchvision import models


if __name__ == '__main__':
    print('hello')
    backbone = models.resnet101(weights='ResNet101_Weights.IMAGENET1K_V1')
