import copy

from bereshit import Object,Camera,Vector3, Rigidbody, BoxCollider, Mesh_rander, FixJoint,Quaternion
from FPS_cam import rotate
from CamController import CamController
from debug import debug
import Core
from Shoot import Shoot

import copy

# base object template
base_obj = Object(
    position=(0, 1, 0),
    size=(1, 1, 1),
    rotation=(0, 0, 0),
    name="obj"
)
base_obj.add_component(Rigidbody(useGravity=True, velocity=Vector3(0, 0, 0), restitution=0.6))
base_obj.add_component(BoxCollider())

# stack of 10 objects
stack = []
for i in range(3):
    obj_copy = copy.deepcopy(base_obj)
    obj_copy.position = Vector3(0, 2+i*1.5, 0)  # space slightly so no initial overlap
    obj_copy.name = f"obj_{i}"
    stack.append(obj_copy)

# floor
floor = Object(position=(0,0,0), size=(1,1,1), rotation=(0,0,0), name="floor")
floor.add_component(Rigidbody(useGravity=False, restitution=0.6,mass=9999))
floor.add_component(BoxCollider())

# camera
camera = Object(size=(1,1,1), position=(0,0,-5), name='camera')
camera.add_component(Camera())
camera.add_component(Mesh_rander(shape="empty"))
camera.add_component(CamController())
camera.add_component(Rigidbody(isKinematic=True))
camera.add_component(BoxCollider())
camera.add_component(rotate())
camera.add_component(Shoot())

# scene with stacked objects
scene = Object(
    position=Vector3(0,0,0),
    size=(0,0,0),
    children=stack + [floor, camera],
    name="scene"
)

Core.run(scene, gizmos=False, speed=1, tick=1/60, scriptRefreshRate=1)
