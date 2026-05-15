import time

from bereshit.Quaternion import Quaternion
from bereshit.Vector3 import Vector3
start = time.process_time()

vec = Vector3(1230,123,5345)

qunt = Quaternion(1650,21364,32245, 123).normalized()

for i in range(10000000):
    a = qunt.rotate(vec)

print(time.process_time() - start)

