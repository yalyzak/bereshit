from bereshit import Object, Vector3, Core, Camera, Rigidbody, Physics
from bereshit.BoxCollider import BoxCollider
from bereshit.addons.essentials import FPS_cam, CamController, Shoot

cam = Object(position=Vector3(5, 0, -2)).add_component(Camera(), CamController(), FPS_cam())

floor = Object(size=Vector3(100,1,100), position=Vector3(0,-1,0), rotation=Vector3(0,0,0)).add_component(BoxCollider(), Rigidbody(isKinematic=True))

obj1 = Object(position=Vector3(0,5,0), size=Vector3(1,1,1)).add_component(BoxCollider(), Rigidbody())

hit = Physics.Raycast(Vector3(0,0,0), Vector3(0,1,0), obj1.Collider)

print(hit.point)




