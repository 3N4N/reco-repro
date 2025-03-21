import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class PascalVOCSegmentation(Dataset):

    def __init__(self, root_dir, split='train', input_size=513):
        self.root_dir = root_dir
        self.split = split
        self.input_size = input_size

        split_file = os.path.join(
            root_dir,
            'ImageSets',
            'Segmentation',
            f'{split}.txt'
        )

        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        img_path = os.path.join(
            self.root_dir,
            'JPEGImages',
            f'{img_id}.jpg'
        )
        img = Image.open(img_path).convert('RGB')

        mask_path = os.path.join(
            self.root_dir,
            'SegmentationClass',
            f'{img_id}.png'
        )
        mask = Image.open(mask_path)

        img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        mask = mask.resize((self.input_size, self.input_size), Image.NEAREST)
        img = TF.to_tensor(img)
        mask = torch.from_numpy(np.array(mask)).long()
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask[mask == 255] = -1

        return img, mask

