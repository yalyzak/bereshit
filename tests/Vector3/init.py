from bereshit import Vector3
import time
start = time.perf_counter()

for i in range(10000000):
    V = Vector3(1,1,1)


print(time.perf_counter() - start)