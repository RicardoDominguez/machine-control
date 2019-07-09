from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


def get_required_argument(dotmap, key, message, default=None):
    """Returns an argument from a dotmap object, raises and error if it does not exist.

    Arguments:
        dotmap (dotmap).
        key (str).
        message (str): Error message to be raised.
    """
    val = dotmap.get(key, default)
    if val is default:
        raise ValueError(message)
    return val
