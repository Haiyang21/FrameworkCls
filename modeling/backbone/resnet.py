import torch
import torchvision.models as models


def resnet18(cfg):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, cfg.MODEL.NUM_CLASS)

    return model

