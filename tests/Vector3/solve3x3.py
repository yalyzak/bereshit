import numpy as np
from bereshit.Joint import Joint
from bereshit import Vector3
import time
start = time.perf_counter()


K = np.array([
    [3.0, 2.0, -1.0],
    [2.0, -2.0, 4.0],
    [-1.0, 0.5, -1.0]
])

b = np.array([1.0, -2.0, 0.0])

for i in range(10000000):
    result = Joint.solve3x3(K, b)

print(time.perf_counter() - start)