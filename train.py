import torch
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchvision.models.segmentation import fcn_resnet50

from data.pascal_voc_dataset import PascalVOCSegmentation
from data.utils import get_pascal_dataloader

LEARNING_RATE = 3e-4
BATCH_SIZE = 32
EPOCHS = 2
DATA_ROOT = "."
num_workers = 0

device = "cuda" if torch.cuda.is_available() else "cpu"

