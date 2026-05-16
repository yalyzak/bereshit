from bereshit import Vector3, Quaternion
from bereshit.Cache import Cache
import time
start = time.perf_counter()
V = Vector3(1,1,1)
qunt = Quaternion(10,213,3245, 123).normalized()
cache = Cache()

for i in range(10000000):
    V.MatrixMultiplication(cache.R)

print(time.perf_counter() - start)


