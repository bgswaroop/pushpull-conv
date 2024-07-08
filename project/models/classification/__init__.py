from .alexnet import AlexNet
from .cct import cct_14_7x2_224
from .convnet import ConvNet
from .convnext import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge
from .efficientnet import EfficientNet
from .resnet import (resnet18, resnet34, resnet50, resnet101, resnet152,
                     resnext50_32x4d, resnext101_32x8d, resnext101_64x4d)
from .resnet_zhang_ICML2019 import resnet50 as zhang_resnet50
from .resnet_vasconcelos_etal import resnet50 as vasconcelos_resnet50
from .resnet_strisciuglio_etal import resnet50 as strisciuglio_resnet50

__all__ = (
    'AlexNet',
    'ConvNet',
    'convnext_tiny',
    'EfficientNet',
    'cct_14_7x2_224',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d',
    'vasconcelos_resnet50', 'zhang_resnet50', 'strisciuglio_resnet50',
    'get_classifier'
)


def efficientnet_wrapper(args):
    net = EfficientNet.from_name(args.model,
                                 lightning_args=args,
                                 use_push_pull=args.use_push_pull,
                                 pp_avg_kernel_size=args.avg_kernel_size,
                                 pull_inhibition=args.pull_inhibition_strength,
                                 trainable_pull_inhibition=args.trainable_pull_inhibition,
                                 num_classes=args.num_classes)
    return net


def convnext_wrapper(args):
    if args.model == 'convnext-tiny':
        net = convnext_tiny(args)
    else:
        raise ValueError('Other ConvNext models need to be implemented')
    return net


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
    elif 'efficientnet' in args.model:
        model = efficientnet_wrapper(args)
    elif 'convnext' in args.model:
        model = convnext_wrapper(args)
    elif 'CCT-ImageNet' in args.model:
        model = cct_14_7x2_224(lightning_args=args)
    elif args.model == 'zhang_resnet50':
        model = zhang_resnet50(args)
    elif args.model == 'vasconcelos_resnet50':
        model = vasconcelos_resnet50(args)
    elif args.model == 'strisciuglio_resnet50':
        model = strisciuglio_resnet50(args)
    else:
        raise ValueError('Invalid classifier')

    return model
