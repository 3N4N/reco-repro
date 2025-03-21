import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class TeacherModel(nn.Module):
    def __init__(self, student_model, ema_decay=0.99):
        super(TeacherModel, self).__init__()
        
        self.model = copy.deepcopy(student_model)
        self.ema_decay = ema_decay
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.model(x)
    
