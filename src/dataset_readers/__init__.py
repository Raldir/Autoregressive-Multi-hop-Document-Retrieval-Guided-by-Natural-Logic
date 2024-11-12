import os

from .fever_reader import FeverDatasetReader
from .feverous_reader import FeverousDatasetReader

def get_class(name):
    if name == 'fever':
        return FeverDatasetReader
    elif name == 'feverous':
        return FeverousDatasetReader

    raise RuntimeError('Invalid dataset: %s' % name)
