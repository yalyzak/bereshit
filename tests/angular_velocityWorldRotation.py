import math
import keyboard

from bereshit import Object, Vector3, Core, Camera, FixedJoint, HingeJoint, BoxCollider, Rigidbody, Quaternion
from bereshit.addons.essentials import FPS_cam, CamController

cam = Object(position=Vector3(0, 0, -8)).add_component(Camera(), CamController())


servo = Object(name="servo", position=Vector3(0,2,0), rotation=Vector3(0,20,0)).add_component(BoxCollider(), Rigidbody())

servo.quaternion *= Quaternion.euler(Vector3(90, 0, 0))

servo2 = Object(name="servo2", position=Vector3(5,2,0)).add_component(BoxCollider(), Rigidbody())


servo.Rigidbody.angular_velocity.z += 1
servo2.Rigidbody.angular_velocity.z += 1

Core.run([cam, servo, servo2])
