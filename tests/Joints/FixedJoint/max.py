import copy

from bereshit import Object, Vector3, Core, Camera, BoxCollider, Rigidbody, FixedJoint, HingeJoint
from bereshit.addons.essentials import FPS_cam, CamController
from bereshit.addons.PPO.examples.WalkToGoal.Servo import Servo

cam = Object(position=Vector3(0, 0, -8)).add_component(Camera(shading="material preview"))

floor = Object(size=Vector3(100, 1, 100), position=Vector3(0, -3, 0)).add_component(BoxCollider(),
                                                                                    Rigidbody(isKinematic=True))


feet = Object(size=Vector3(1, 1, 1), position=Vector3(0,0,0), name="feet").add_component(BoxCollider(), Rigidbody(isKinematic=True))

mount = Object(position=Vector3(2,0,0), size=Vector3(1, 1, 1), name="mount").add_component(BoxCollider(), Rigidbody(mass=1), FixedJoint(feet, beta=1))

weight = Object(position=Vector3(2,2,0)).add_component(BoxCollider(), Rigidbody(mass=100))

Core.run([cam, mount, feet, weight], Render=True, tick=1/120, physics_epochs=1)
