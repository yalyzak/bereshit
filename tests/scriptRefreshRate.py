from bereshit import Object, Vector3, Core, Camera, BoxCollider, Rigidbody
from bereshit.addons.essentials import FPS_cam, CamController

class Debug:
    def __init__(self):
        self.i = 0
        self.dt = 0
    def Update(self, dt):
        self.i += 1
        self.dt += dt
        if self.dt % 1/60 == 0:
            print("pass")
        print(self.i)
cam = Object(position=Vector3(0, 0, -8)).add_component(Camera(), CamController(), FPS_cam(), Debug())

floor = Object(size=Vector3(10,1,10), position=Vector3(0,-1,0)).add_component(BoxCollider(), Rigidbody(isKinematic=True))

obj2 = Object(position=Vector3(0,2,0)).add_component(BoxCollider(), Rigidbody())

Core.run([cam, floor, obj2], MaxTime=10, speed=10)
