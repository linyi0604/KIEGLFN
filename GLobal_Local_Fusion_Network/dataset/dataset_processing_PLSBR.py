import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np


class DatasetProcessing(Dataset):
    def __init__(self, data_path, img_filename, transform=None, feature_type=None):
        self.img_path = data_path
        self.transform = transform
        self.feature_type = feature_type
        # reading img file from file
        img_filepath = img_filename
        fp = open(img_filepath, 'r')

        self.img_filename = []
        self.labels = []
        self.lesions = []
        for line in fp.readlines():
            filename, label = line.split()
            self.img_filename.append(filename)
            self.labels.append(int(label))
            # self.lesions.append(int(lesion))
        # self.img_filename = [x.strip() for x in fp]
        fp.close()
        self.img_filename = np.array(self.img_filename)
        self.labels = np.array(self.labels)#.reshape(-1, 1)
        self.lesions = np.array(self.lesions)#.reshape(-1, 1)

        if 'train' in img_filename:
            ratio = 1.0#0.1
            import random
            random.seed(42)
            indexes = []
            for i in range(4):
                index = random.sample(list(np.where(self.labels == i)[0]), int(len(np.where(self.labels == i)[0]) * ratio))
                indexes.extend(index)
            self.img_filename = self.img_filename[indexes]
            self.labels = self.labels[indexes]
            # self.lesions = self.lesions[indexes]


    def get_skin_feature_torch(self, img):
        color = []

        _, w, h = img.shape
        wmid = w//2
        hmid = h//2

        crop = img[:, wmid-1: wmid+2, hmid-1: hmid+2]

        for i in range(3):
            for j in range(3):
                r, g, b = crop[:, i, j]
                color.append((r, g, b))
        if self.feature_type == "max":
            skin_feature = np.max(color, axis=0)
        elif self.feature_type == "min":
            skin_feature = np.min(color, axis=0)
        elif self.feature_type == "median":
            skin_feature = np.median(color, axis=0)
        elif self.feature_type in ["patch", "mapping"]:
            skin_feature = crop
        elif self.feature_type == "mean":
            skin_feature = np.mean(color, axis=0)
        else:
            raise
        skin_feature = torch.FloatTensor(skin_feature)
        return skin_feature

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        # skin_color = self.get_skin_color_mean(img)

        if self.transform is not None:
            img = self.transform(img)
        name = self.img_filename[index]
        label = torch.from_numpy(np.array(self.labels[index]))
        skin_feature = self.get_skin_feature_torch(img)

        return img, label, skin_feature

    def __len__(self):
        return len(self.img_filename)
