import time
from bereshit.Quaternion import Quaternion
start = time.process_time()

qunt = Quaternion(10,213,3245, 123).normalized()


for i in range(10000000):
    qunt.to_matrix3()

print(time.process_time() - start)

