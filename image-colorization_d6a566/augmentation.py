from numpy.random.mtrand import rand, shuffle
import torchvision.transforms as T
import torch
import torchvision
import matplotlib.pyplot as plt
import glob
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import numpy as np
from torchvision.transforms.functional import center_crop

class ImageDataset(Dataset):
    def __init__(self,image_list,transforms_list=None):
        self.image_list=image_list
        self.transforms_list=transforms_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,i):
        img = Image.open(self.image_list[i])
        out_list = [img]
        
        if not self.transforms_list == None:   
            for transform in self.transforms_list:
                if transform == 'rotation':
                    random_rotation = torchvision.transforms.RandomRotation(degrees = np.random.randint(10,45))
                    out_list.append(random_rotation(img))
                elif transform == 'crop':
                    try:
                        random_crop = T.transforms.RandomCrop((256, 256), fill=0, padding_mode='constant')
                        out_list.append(random_crop(img))
                    except:
                        pass
                elif transform == 'flip':
                    horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
                    out_list.append(horizontal_flip(img))

                    vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1)
                    out_list.append(vertical_flip(img))

        return out_list
