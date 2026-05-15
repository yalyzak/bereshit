from bereshit import Vector3, Quaternion
import time
start = time.perf_counter()
V = Vector3(1,1,1)
qunt = Quaternion(10,213,3245, 123).normalized()
R = qunt.to_matrix3()
for i in range(10000000):
    V.MatrixMultiplication(R)

print(time.perf_counter() - start)


