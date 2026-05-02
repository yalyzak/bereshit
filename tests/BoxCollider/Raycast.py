from bereshit import Object, Vector3, Core, Camera, Rigidbody, Physics
from bereshit.BoxCollider2 import BoxCollider as BoxCollider2
from bereshit.addons.essentials import FPS_cam, CamController, Shoot


obj2 = Object(position=Vector3(0,0,0), rotation=Vector3(0,0,0)).add_component(BoxCollider2(), Rigidbody(friction_coefficient=0))

hit = Physics.Raycast(Vector3(0,0,5), Vector3(0,0,-1), obj2.Collider)

print(hit.point)






