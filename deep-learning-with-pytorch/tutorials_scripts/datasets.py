import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io, transform
import torch
from torch.utils.data import DataLoader, Dataset


class FaceLandmarksDataset(Dataset):
    '''
    Implement custom Dataset class to get images and its corresponding
    landmarks.
    '''

    def __init__(self, file_name, root_dir, transform=None):
        self.landmarks_df = pd.read_csv(file_name)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.landmarks_df)
    
    def __getitem__(self, index):
        img_name = self.landmarks_df.iloc[index, 0]
        img = io.imread(os.path.join(self.root_dir, img_name))
        landmarks = self.landmarks_df.iloc[index, 1:].values
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': img, 'landmarks': landmarks}
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample


class Rescale:
    '''Resize an image to a give output size.'''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, (tuple, list)):
            assert len(output_size) == 2
            self.new_h, self.new_w = output_size
        else:
            self.new_h, self.new_w = (output_size, output_size)
    
    def __call__(self, sample):
        img, landmarks = sample['image'], sample['landmarks']
        h, w = img.shape[:2]
    
        img = transform.resize(img, (self.new_h, self.new_w))
        landmarks = landmarks * [self.new_w / w, self.new_h / h]
        sample = {'image': img, 'landmarks': landmarks}

        return sample


class RandomCrop:
    '''Randomly crop an image to a given output size.'''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, (tuple, list)):
            assert len(output_size) == 2
            self.new_h, self.new_w = output_size
        else:
            self.new_h, self.new_w = (output_size, output_size)
    
    def __call__(self, sample):
        img, landmarks = sample['image'], sample['landmarks']
        h, w = img.shape[:2]

        top_left = np.random.randint(0, h - self.new_h)
        bottom_left = np.random.randint(0, w - self.new_w)
        img = img[top_left:(top_left + self.new_h), bottom_left:(bottom_left + self.new_w)]
        landmarks = landmarks - [bottom_left, top_left]
        sample = {'image': img, 'landmarks': landmarks}

        return sample


class ToTensor:
    '''Convert a given image from numpy array to tensor.'''

    def __call__(self, sample):
        img, landmarks = sample['image'], sample['landmarks']
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)
        landmarks = torch.from_numpy(landmarks)
        sample = {'image': img, 'landmarks': landmarks}

        return sample
