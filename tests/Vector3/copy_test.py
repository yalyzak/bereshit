import copy_test

from bereshit import Vector3
import time

start = time.perf_counter()
V = Vector3(1, 1, 1)

for i in range(10000000):
    v2 = V.copy()

print(time.perf_counter() - start)


start = time.perf_counter()
V = Vector3(1, 1, 1)

for i in range(10000000):
    v2 = copy.copy(V)

print(time.perf_counter() - start)

