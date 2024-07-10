import sys
import time

import numpy as np

def angle_between(a, b):
    ax = (1/np.linalg.norm(a))*a
    bx = (1/np.linalg.norm(b))*b
    
    d = ax@bx
    phi = np.arccos(d)
    
    return phi

