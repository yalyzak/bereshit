import numpy as np


class Cache:
    def __init__(self):
        self.R = np.empty((3, 3), dtype=np.float64)
        self.R_abs = np.empty((3, 3), dtype=np.float64)
        self.skewA = np.zeros((3, 3), dtype=np.float64)
        self.skewB = np.zeros((3, 3), dtype=np.float64)
        self.rotation_dirty = True
        self.rotation_dirty_abs = True
        self.aabb_dirty = True

    def set_dirty(self):
        self.rotation_dirty = True
        self.rotation_dirty_abs = True
        self.aabb_dirty = True
