from bereshit import Vector3
import time
start = time.perf_counter()
V = Vector3(1,1,1)
U = Vector3(10,0,5)

for i in range(10000000):
    V.cross(U)

print(time.perf_counter() - start)