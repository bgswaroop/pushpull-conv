from . import classification
from . import retrieval

__all__ = ('get_classifier',)


def get_classifier(task, model):
    if task == 'classification':
        return classification.get_classifier(model)
    elif task == 'retrieval':
        return retrieval.get_classifier(model)
    else:
        raise ValueError('Invalid task!')