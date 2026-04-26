from bereshit import Object, Vector3, Core, Camera, BoxCollider, Rigidbody
from bereshit.addons.essentials import FPS_cam, CamController

cam = Object(position=Vector3(0,-4,-3)).add_component(Camera(shading="material preview"), CamController(), FPS_cam())

obj1 = Object(position=Vector3(2,-4,0)).add_component(BoxCollider(), Rigidbody(velocity=Vector3(10,0,0), Freeze_Rotation=Vector3(1,1,1)))

obj2 = Object(position=Vector3(0,-4,0)).add_component(BoxCollider(), Rigidbody(angular_velocity=Vector3(0,1,0)))

obj3 = Object(position=Vector3(3,2,0)).add_component(BoxCollider(), Rigidbody(velocity=Vector3(1,0,0), angular_velocity=Vector3(0,1,0), restitution=0.1))

floor = Object(size=Vector3(100,1,100), position=Vector3(0,-5,0)).add_component(BoxCollider(), Rigidbody(isKinematic=True))

Core.run([cam,floor, obj1, obj2, obj3])
