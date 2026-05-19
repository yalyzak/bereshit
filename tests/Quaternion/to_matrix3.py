import time
from bereshit.Quaternion import Quaternion
from bereshit.Cache import Cache

start = time.process_time()

qunt = Quaternion(10,213,3245, 123).normalized()

cache = Cache()

for i in range(10000000):
    qunt.to_matrix3(cache)

print(time.process_time() - start)

