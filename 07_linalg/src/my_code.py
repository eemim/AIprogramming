import sys
import time
import numpy as np

def unit(a):
    x = np.linalg.norm(a)
    hat_a=(1/x)*a
    return hat_a
