import math
import keyboard

from bereshit import Object, Vector3, Core, Camera, FixedJoint, HingeJoint, BoxCollider, Rigidbody, Quaternion
from bereshit.addons.essentials import FPS_cam, CamController

cam = Object(position=Vector3(0, 1, -2)).add_component(Camera(), CamController())


servo = Object(name="servo", position=Vector3(0,2,0), rotation=Vector3(0,0,10)).add_component(BoxCollider(), Rigidbody())

# servo2 = Object(name="servo2", position=Vector3(5,2,0)).add_component(BoxCollider(), Rigidbody())

floor = Object(size=Vector3(10,1,10)).add_component(BoxCollider(), Rigidbody(isKinematic=True))


Core.run([cam, servo, floor])
