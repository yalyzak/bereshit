class debug:
    def OnTriggerEnter(self, Collision):
        print("entered", Collision.other.parent.name)

    def OnTriggerStay(self, Collision):
        print("stay", Collision.other.parent.name)

    def OnTriggerExit(self, Collision):
        print("exited", Collision.other.parent.name)

from bereshit import Object, Vector3, Core, Camera, BoxCollider, Rigidbody
from bereshit.addons.essentials import FPS_cam, CamController

cam = Object(position=Vector3(0, 0, -8)).add_component(Camera(), CamController(), FPS_cam())

floor = Object(size=Vector3(10, 1, 10), position=Vector3(0, -1, 0), name="floor").add_component(BoxCollider(is_trigger=True),
                                                                                  Rigidbody(isKinematic=True), debug())

obj = Object(position=Vector3(0, 2, 0), name="obj").add_component(BoxCollider(), Rigidbody())

Core.run([cam, floor, obj])
