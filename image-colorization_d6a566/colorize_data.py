from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import os

from augmentation import *

from torchvision.transforms.functional import resize

class ColorizeData(Dataset):
    def __init__(self,image_path):
        # Initialize dataset, you may use a second dataset for validation if required
        self.image_path = image_path
        self.image_list = glob.glob(image_path+'/*.jpg')

        # Use the input transform to convert images to grayscale
        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256,256)),
                                          T.Grayscale(),
                                          T.Normalize((0.5), (0.5))
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.ToTensor(),
                                           T.Resize(size=(256,256)),
                                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    

    def __len__(self) -> int:
        # return Length of dataset
        return len(self.image_list)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return the input tensor and output tensor for training
        img = plt.imread(self.image_list[index])
        img=Image.fromarray(img).convert('RGB')
        img=np.array(img).astype(np.uint8)
        gray_img = self.input_transform(img)
        gray_img = torch.tensor(gray_img,dtype=torch.float)
        target_img = self.target_transform(img)
        target_img = torch.tensor(target_img,dtype=torch.float)
        return (gray_img,target_img)
        