from torchvision.transforms import functional as FT
from torchvision import transforms
import numpy as np
import random
import cv2
from PIL import Image


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image):
        if self.to_bgr255:
            image = image * 255
            # image = image[[2,1,0]] * 255
        image = FT.normalize(image, mean=self.mean, std=self.std)
        return image


class Resize(object):
    def __init__(self, scale_w, scale_h, to_pil_format=False):
        self.scale_w = scale_w
        self.scale_h = scale_h
        self.to_pil_format = to_pil_format

    def __call__(self, image):
        img_resize = image
        if image.shape[0] != self.scale_h or image.shape[1] != self.scale_w:
            img_resize = cv2.resize(image, (self.scale_w, self.scale_h))
        if self.to_pil_format:
            return Image.fromarray(img_resize)
        else:
            return img_resize


class ResizePad(object):
    def __init__(self, scale_w, scale_h, to_pil_format=False):
        self.scale_w = scale_w
        self.scale_h = scale_h
        self.to_pil_format = to_pil_format

    def __call__(self, image):
        img_pad = np.zeros((self.scale_h, self.scale_w, 3), dtype='uint8')
        height, width, channel = image.shape
        max_len = max(height, width)
        max_len_scale = max(self.scale_h, self.scale_w)

        fc = float(max_len_scale) / float(max_len)
        resize_w = int(width * fc)
        resize_h = int(height * fc)
        img_resize = cv2.resize(image, (resize_w, resize_h))
        img_pad[0:resize_h, 0:resize_w] = img_resize

        if self.to_pil_format:
            return Image.fromarray(img_pad)
        else:
            return img_pad


class RandomCropWithBBox(object):
    def __init__(self, scale_list):
        self.scale_list = scale_list

    def __call__(self, image, bbox):
        height, width, channel = image.shape
        fc = random.choice(self.scale_list)
        w_b = bbox[2] - bbox[0]
        h_b = bbox[3] - bbox[1]
        x0_ext = int(max(0, bbox[0] - w_b/fc))
        y0_ext = int(max(0, bbox[1] - h_b/fc))
        x1_ext = int(min(width-1, bbox[2] + w_b/fc))
        y1_ext = int(min(height-1, bbox[3] + h_b/fc))
        img_crop = image[y0_ext:y1_ext, x0_ext:x1_ext]
        return img_crop

