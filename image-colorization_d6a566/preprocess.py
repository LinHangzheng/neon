import os
import glob
import numpy as np
from augmentation import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from  options import parse_options

class preprocess():
    def __init__(self,args) -> None:
        self.args = args
        # create data path for training and testing
        self._build_path()
        # get the images of training and testing
        self._get_image()
        self._get_dataset()

    def _build_path(self):
        if not os.path.exists(args.train_path):
            os.makedirs(args.train_path)
        if not os.path.exists(args.test_path):
            os.makedirs(args.test_path)

    def _get_image(self):
        # get the image path
        img_path = self.args.img_path
        img_list = glob.glob(img_path+'/*.jpg')

        # get the training images and testing images
        img_length = len(img_list)
        train_size= int(self.args.train_test_ratio*img_length)

        np.random.shuffle(img_list)
        self.train_img = img_list[:train_size]
        self.test_img = img_list[train_size:]

    def _get_dataset(self):
        transforms = self.args.transforms
        self.train_dataset = ImageDataset(self.train_img, transforms)
        self.test_dataset = ImageDataset(self.test_img)

    def export_imgs(self):
        for img_num, data in enumerate(tqdm(self.train_dataset)):
            for idx, img in enumerate(data):
                img.save(self.args.train_path+str(img_num)+'_'+str(idx)+'.jpg')

        for img_num, data in enumerate(tqdm(self.test_dataset)):
            img = data[0]
            img.save(self.args.test_path+str(img_num)+'.jpg')


if __name__ == '__main__':
    args, args_str = parse_options()
    preprocess = preprocess(args)
    preprocess.export_imgs()

    






