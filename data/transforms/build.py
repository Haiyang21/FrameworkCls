from data.transforms.transforms import *
from torchvision import transforms


def generalized_transform(cfg, phase):
    mean = cfg.DATASET.MEAN
    std = cfg.DATASET.STD
    normalize_transform = Normalize(mean=mean, std=std, to_bgr255=True)
    resize_shape = cfg.DATASET.RESIZE_SHAPE
    crop_shape = cfg.DATASET.CROP_SHAPE
    resize_transform = Resize(resize_shape[0], resize_shape[1], to_pil_format=True)

    if phase == 'train':
        data_transforms = transforms.Compose([
            resize_transform,
            transforms.RandomCrop(crop_shape[0]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-5, 5)),
            transforms.ToTensor(),
            normalize_transform
        ])
    else:
        data_transforms = transforms.Compose([
            resize_transform,
            transforms.CenterCrop((crop_shape[0], crop_shape[1])),
            transforms.ToTensor(),
            normalize_transform
        ])

    return data_transforms


META_TRANSFORMS = {
    'generalized_transform': generalized_transform,
}


def build_transforms(cfg, phase):
    meta_transform = META_TRANSFORMS[cfg.DATASET.TRANSFORM_NAME]
    return meta_transform(cfg, phase)
