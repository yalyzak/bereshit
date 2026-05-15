import time

from bereshit.Quaternion import Quaternion
from bereshit.Vector3 import Vector3
start = time.process_time()

vec = Vector3(1230,123,5345)

vec2 = Vector3(11230,123,45345)

qunt = Quaternion(10,-213,3245, 123).normalized()

qunt2 = Quaternion(1650,21364,32245, 123).normalized()

for i in range(10000000):
    a = qunt.rotate_conjugated(vec)

print(time.process_time() - start)

