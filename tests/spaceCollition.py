from bereshit import Object, Vector3, Core, Camera, FixedJoint, HingeJoint, BoxCollider, Rigidbody, Quaternion
from bereshit.addons.essentials import FPS_cam, CamController

cam = Object(position=Vector3(0, 1, -10)).add_component(Camera(), CamController())


obj1 = Object(name="obj1", position=Vector3(0,2.7,0), rotation=Vector3(0,0,0)).add_component(BoxCollider(), Rigidbody(restitution=1))

obj1.Rigidbody.velocity += Vector3(2,0,0)

obj2 = Object(name="obj2", position=Vector3(5,2,0)).add_component(BoxCollider(), Rigidbody(restitution=1))


Core.run([cam, obj1, obj2], gravity=Vector3())
