from .alexnet import AlexNet
from .convnet import ConvNet
from .resnet import (resnet18, resnet34, resnet50, resnet101, resnet152,
                     resnext50_32x4d, resnext101_32x8d, resnext101_64x4d)

__all__ = (
    'AlexNet', 'ConvNet',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d',
    'get_classifier'
)


def get_classifier(model):
    if model == 'ConvNet':
        Classifier = ConvNet
    elif model == 'AlexNet':
        Classifier = AlexNet
    elif model == 'resnet18':
        Classifier = resnet18
    elif model == 'resnet34':
        Classifier = resnet34
    elif model == 'resnet50':
        Classifier = resnet50
    else:
        raise ValueError('Invalid classifier')
    return Classifier
