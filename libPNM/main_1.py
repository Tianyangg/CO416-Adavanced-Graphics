import numpy as np
from PNM import *

from matplotlib import pyplot as mp
import numpy as np

mu = 0.5
sig = 0.15
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
x_values = np.linspace(0, 5, 120)
mp.plot(x_values, gaussian(x_values, mu, sig))
print(gaussian(1, mu, sig))
print(np.multiply([1,2,3], [1,4,9]))
a = np.array([[1,2,3],[4,5,6]])
b = np.array([1,2,3])

print(np.divide(a, b))
"""
part1: assemble pmf file to hdr image
"""

def HDR_Assemble():
    exposures_number = 7
    Zi = [] #
    pass
