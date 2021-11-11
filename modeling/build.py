from modeling.backbone.resnet import resnet18
from modeling.backbone.resnet_face18 import resnet_face18

META_ARCHITECTURES = {
    "resnet18": resnet18,
    "resnet_face18": resnet_face18,
}


def build_model(cfg):
    meta_arch = META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)

