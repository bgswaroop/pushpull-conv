from .alexnet import AlexNet
from .convnet import ConvNet
from .resnet import (resnet18, resnet34, resnet50, resnet101, resnet152,
                     resnext50_32x4d, resnext101_32x8d, resnext101_64x4d)
from .resnet_zhang_ICML2019 import resnet50 as zhang_resnet50
from .resnet_vasconcelos_etal import resnet50 as vasconcelos_resnet50
from .resnet_strisciuglio_etal import resnet50 as strisciuglio_resnet50

__all__ = (
    'AlexNet', 'ConvNet',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d',
    'vasconcelos_resnet50', 'zhang_resnet50', 'strisciuglio_resnet50',
    'get_classifier'
)


def get_classifier(args):
    if args.model == 'ConvNet':
        model = ConvNet(args)
    elif args.model == 'AlexNet':
        model = AlexNet(args)
    elif args.model == 'resnet18':
        model = resnet18(args)
    elif args.model == 'resnet34':
        model = resnet34(args)
    elif args.model == 'resnet50':
        model = resnet50(args)
    elif args.model == 'zhang_resnet50':
        model = zhang_resnet50(args)
    elif args.model == 'vasconcelos_resnet50':
        model = vasconcelos_resnet50(args)
    elif args.model == 'strisciuglio_resnet50':
        model = strisciuglio_resnet50(args)
    else:
        raise ValueError('Invalid classifier')

    return model
