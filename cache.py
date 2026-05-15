import numpy as np


class Cache:
    def __init__(self):
        self.R = np.empty((3, 3), dtype=np.float64)