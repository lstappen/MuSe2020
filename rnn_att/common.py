import os
from collections import namedtuple


def make_dirs_safe(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def dict_to_struct(obj):
    obj = namedtuple("Configuration", obj.keys())(*obj.values())
    return obj
