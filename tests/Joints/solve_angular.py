import time

from bereshit import Object, Vector3, Core, Camera, BoxCollider, Rigidbody, FixedJoint, HingeJoint, World
from bereshit.addons.essentials import FPS_cam, CamController
start = time.process_time()


cam = Object(position=Vector3(0, 0, -8)).add_component(Camera(shading="material preview"))


mount2 = Object(position=Vector3(0,8,0), size=Vector3(.5, .5, .5)).add_component(BoxCollider(), Rigidbody(mass=0.01))

knee = Object(position=Vector3(0,9,0), size=Vector3(.5, .5, .5)).add_component(BoxCollider(), Rigidbody(mass=0.01), HingeJoint(mount2, axis=Vector3(0,0,1)))

joint = knee.get_component(HingeJoint)

for i in range(1000000):
    joint.solve_angular(1/60)


print(time.process_time() - start) # 11.7

