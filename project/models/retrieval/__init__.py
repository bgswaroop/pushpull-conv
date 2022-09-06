from .resnet import (resnet18, resnet34, resnet50, resnet101, resnet152,
                     resnext50_32x4d, resnext101_32x8d, resnext101_64x4d)

__all__ = (
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d',
    'get_classifier'
)


def get_classifier(args):
    if args.model == 'resnet18':
        model = resnet18(args)
    elif args.model == 'resnet34':
        model = resnet34(args)
    elif args.model == 'resnet50':
        model = resnet50(args)
    else:
        raise ValueError('Invalid classifier')

    return model
