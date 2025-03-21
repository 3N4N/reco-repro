import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as F

class CityScapesDataset(Dataset):
    def __init__(self, data_dir, mode='train', target_size=(512, 1024)):
        self.data_dir = data_dir
        self.mode = mode
        self.target_size = target_size
        
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, f'images/{mode}/*.png')))
        
        self.id_to_trainid = {
            7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
            23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
        }
        
        self.background_ids = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30]
    
    def __len__(self):
        return len(self.image_paths)
    
    def _map_labels(self, label_array):
        height, width = label_array.shape
        mapped_labels = np.ones((height, width), dtype=np.uint8) * 255
        
        for orig_id, train_id in self.id_to_trainid.items():
            mapped_labels[label_array == orig_id] = train_id
            
        return mapped_labels
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        img_name = os.path.basename(img_path)
        label_path = os.path.join(self.data_dir, f'labels/{self.mode}/{img_name}')
        
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)
        
        image = image.resize(self.target_size[::-1], Image.BILINEAR)
        label = label.resize(self.target_size[::-1], Image.NEAREST)
        
        image_array = np.array(image)
        label_array = np.array(label)
        
        mapped_label = self._map_labels(label_array)
        
        image_tensor = F.to_tensor(image)
        label_tensor = torch.from_numpy(mapped_label).long()
        
        image_tensor = F.normalize(
            image_tensor, 
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        return image_tensor, label_tensor