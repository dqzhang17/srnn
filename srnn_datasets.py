from __future__ import print_function, division
import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from PIL import Image

class SrnnDataset(Dataset):
    def __init__(self, root_dir, split, model_config):
        self.root_dir = root_dir
        self.image_folder = datasets.ImageFolder(root=root_dir, transform=None)
        self.split = split
        # model specific
        self.mean = model_config.mean
        self.std = model_config.std
        self.input_size = max(model_config.input_size) # nn input size
        self.scale_size = int(math.floor(self.input_size/0.875))   # for input=224, scale_size=256

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        image, target = self.image_folder[idx]

        scaled_images = []
        if self.split=='train':
            img_transform1 = transforms.Compose([
                transforms.RandomSizedCrop(448),
                transforms.RandomHorizontalFlip()])
            scaled_images += [img_transform1(image)]
            scaled_images += [transforms.Scale(self.input_size, Image.BICUBIC)(scaled_images[0])]
        else:
            img_transform1 = transforms.Compose([transforms.Scale(448, Image.BICUBIC),transforms.CenterCrop(448)])
            img_transform2 = transforms.Compose([transforms.Scale(self.scale_size, Image.BICUBIC),transforms.CenterCrop(self.input_size)])
            scaled_images = [img_transform1(image), img_transform2(image)]

        transform_totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean,
                             std=self.std)
        ])
        scaled_images = [transform_totensor(img) for img in scaled_images]
        scaled_images.reverse()

#        always put smaller images first
        sample = {'image': scaled_images, 'target': target}

        return sample

# test
if __name__ == "__main__":
    pass
