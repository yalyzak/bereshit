import copy

from bereshit import Object, Vector3, Core, Camera, BoxCollider, Rigidbody
from bereshit.addons.essentials import FPS_cam, CamController

cam = Object(position=Vector3(0, 0, -8)).add_component(Camera(shading="material preview"), CamController(), FPS_cam())

floor = Object(size=Vector3(1000,1,1000), position=Vector3(0,-1,0)).add_component(BoxCollider(), Rigidbody(isKinematic=True))

obj1 = Object(position=Vector3(0,2,0)).add_component(BoxCollider(), Rigidbody())
objs = []
for i in range(100):
    obj2 = copy.deepcopy(obj1)
    obj2.position += Vector3.random() * 100
    objs.append(obj2)

Core.run([floor, cam] + objs, tick=1/60, Render=False, speed=10, MaxTime=5)
