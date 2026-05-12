from bereshit import Vector3
import time
start = time.perf_counter()
V = Vector3(1,1,1)
U = Vector3(10,0,5)

for i in range(999999999):
    V.dot(U)

print(time.perf_counter() - start)