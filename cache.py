import numpy as np


class Cache:
    def __init__(self):
        self.R = np.empty((3, 3), dtype=np.float64)
        self.skewA = np.zeros((3, 3), dtype=np.float64)
        self.skewB = np.zeros((3, 3), dtype=np.float64)

# 234234