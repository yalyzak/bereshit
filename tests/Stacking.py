from bereshit import Object, Vector3, Core, Camera, BoxCollider, Rigidbody
from bereshit.addons.essentials import FPS_cam, CamController

cam = Object(position=Vector3(0, 0, -8)).add_component(Camera(), CamController(), FPS_cam())

floor = Object(size=Vector3(10,1,10), position=Vector3(0,-1,0)).add_component(BoxCollider(), Rigidbody(isKinematic=True))

# obj = Object(position=Vector3(0,0,0)).add_component(BoxCollider(), Rigidbody())
#
obj2 = Object(position=Vector3(0,2,0)).add_component(BoxCollider(), Rigidbody())

Core.run([cam, floor, obj2])


