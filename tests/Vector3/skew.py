from bereshit import Vector3
from bereshit.Cache import Cache
import time
start = time.perf_counter()
V = Vector3(1,1,1)
cache = Cache()
for i in range(10000000):
    V.skew(cache.skewA)

print(time.perf_counter() - start)