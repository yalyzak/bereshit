import time
from bereshit import Object, Vector3, BoxCollider, Collider
start = time.process_time()

obj1 = Object(position=Vector3(0,8,0), size=Vector3(.5, .5, .5)).add_component(BoxCollider())

collider = obj1.get_component(BoxCollider)

for i in range(1000000):
    k = collider.get_aabb()

print(time.process_time() - start)