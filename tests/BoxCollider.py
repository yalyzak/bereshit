from bereshit import Object, Vector3, Core, Camera, Rigidbody
from bereshit.BoxCollider2 import BoxCollider as BoxCollider2
from bereshit.BoxCollider import BoxCollider as BoxCollider
from bereshit.addons.essentials import FPS_cam, CamController, Shoot

cam = Object(position=Vector3(0, 0, -2)).add_component(Camera(), CamController(), FPS_cam(), Shoot())

floor = Object(size=Vector3(10,1,10), position=Vector3(0,-1,0), rotation=Vector3(0,0,0)).add_component(BoxCollider2(), Rigidbody(isKinematic=True))

obj1 = Object(position=Vector3(0,0,0), size=Vector3(1,1,1)).add_component(BoxCollider2(), Rigidbody(friction_coefficient=0))

obj2 = Object(position=Vector3(0,0.9,0), size=Vector3(1,1,1), rotation=Vector3(90,0,0)).add_component(BoxCollider2(), Rigidbody(friction_coefficient=0))



Core.run([floor, cam, obj2], gizmos=True)


