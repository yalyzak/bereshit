from bereshit import Vector3
import time
start = time.perf_counter()
V = Vector3(1,1,1)

for i in range(10000000):
    V.skew()

print(time.perf_counter() - start)