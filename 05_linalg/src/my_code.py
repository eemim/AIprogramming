import sys
import time
import numpy as np


def is_orthogonal(a, b):
    if np.sum(a * b) == 0:
        return True
    return False


