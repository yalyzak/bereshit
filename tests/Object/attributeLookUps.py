import time

from bereshit import Object, Vector3, BoxCollider, Rigidbody, HingeJoint
start = time.process_time()

obj1 = Object(position=Vector3(0,8,0), size=Vector3(.5, .5, .5)).add_component(BoxCollider(), Rigidbody(mass=0.01))

vec = Vector3(1230,123,5345)

obj2 = Object(position=Vector3(0,9,0), size=Vector3(.5, .5, .5)).add_component(BoxCollider(), Rigidbody(mass=0.01), HingeJoint(obj1, axis=Vector3(0,0,1)))

joint = obj2.get_component(HingeJoint)

for i in range(1000000):
    k = joint.parent.quaternion.rotate(vec)

print(time.process_time() - start)