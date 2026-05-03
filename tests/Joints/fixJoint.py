from bereshit import Object, Vector3, Core, Camera, BoxCollider, Rigidbody, FixedJoint, HingeJoint
from bereshit.addons.essentials import FPS_cam, CamController

cam = Object(position=Vector3(0, 0, -8)).add_component(Camera(shading="material preview"), CamController(), FPS_cam())


mount2 = Object(position=Vector3(0,8,0), size=Vector3(.5, .5, .5)).add_component(BoxCollider(), Rigidbody(mass=0.01))

knee = Object(position=Vector3(0,9,0), size=Vector3(.5, .5, .5)).add_component(BoxCollider(), Rigidbody(mass=0.01), HingeJoint(mount2, axis=Vector3(0,0,1)))


Core.run([cam, mount2, knee])



