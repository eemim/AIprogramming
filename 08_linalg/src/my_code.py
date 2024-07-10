import sys
import time
import numpy as np

def project(x, y):
    pyx = ((x.dot(y))/(y.dot(y)))*y
    return pyx
