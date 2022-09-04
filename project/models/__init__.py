from . import classification
from . import retrieval

__all__ = ('get_classifier',)


def get_classifier(args):

    if args.task == 'classification':
        return classification.get_classifier(args)
    elif args.task == 'retrieval':
        return retrieval.get_classifier(args.model)
    else:
        raise ValueError('Invalid task!')