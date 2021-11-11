import cv2
import os
import torch
import numpy as np
import random


cls_map = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'surprise': 3,
    'fear': 4,
    'disgust': 5,
    'anger': 6,
}


class ExpressionDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, phase, transform=None):
        self.root_path = cfg.DATASET.ROOT_PATH
        self.phase = phase
        self.transform = transform
        self.file_paths = list()
        self.label = list()
        self.cls_map = cls_map

        if phase == 'train':
            fname = cfg.DATASET.TRAIN_FILE
        elif phase == 'val':
            fname = cfg.DATASET.VAL_FILE
        elif phase == 'test':
            fname = cfg.DATASET.TEST_FILE
        else:
            raise Exception('No dataset')
        with open(fname) as fp:
            lines = fp.readlines()

        for line in lines:
            if phase == 'test':
                img_path, label = line.strip().split(' ')[0], None
            else:
                img_path, label = line.strip().split(' ')
                if label not in self.cls_map.keys():
                    continue
            img_path = os.path.splitext(img_path)[0]
            self.file_paths.append(os.path.join(self.root_path, img_path + '.jpg'))
            self.label.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path, 0)
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)

        if label is not None:
            label = self.cls_map[label]

        return image, label, path
