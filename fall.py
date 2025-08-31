import copy
import random

from bereshit import Object,Camera,Vector3, Rigidbody, BoxCollider, Mesh_rander, FixJoint,Quaternion,Material
from FPS_cam import rotate
from CamController import CamController
from debug import debug
import Core
from Shoot import Shoot

import copy
# base object template
base_obj = Object(
    position=(0, 100, 0),
    size=(5, 1, 2),
    rotation=(0,0, 30),
    name="obj"
)
r=1
base_obj.add_component(Rigidbody(useGravity=True, velocity=Vector3(0, 0, 0), restitution=r))
base_obj.add_component(BoxCollider())
base_obj.add_component(Material(color="blue"))

# stack of 10 objects
stack = []
for i in range(1):
    obj_copy = copy.deepcopy(base_obj)
    obj_copy.position = Vector3(5, 5, 0)  # space slightly so no initial overlap
    obj_copy.name = f"obj_{i}"
    stack.append(obj_copy)

# floor
floor = Object(position=(0,0,1), size=(10,1,10), rotation=(0,0,0), name="floor")
floor.add_component(Rigidbody(isKinematic=True, restitution=r,mass=9999))
floor.add_component(BoxCollider())

# camera
camera = Object(size=(1,1,1), position=(0,0,-5), name='camera')
camera.add_component(Camera(shading="wire"))
camera.add_component(Mesh_rander(shape="empty"))
camera.add_component(CamController())
camera.add_component(Rigidbody(isKinematic=True))
camera.add_component(BoxCollider())
camera.add_component(rotate())
# camera.add_component(Shoot())
# stack[0].add_component(debug(None))
# scene with stacked objects
scene = Object(
    position=Vector3(0,0,0),
    size=(0,0,0),
    children=stack + [floor, camera],
    name="scene"
)

Core.run(scene, gizmos=True, speed=.5, tick=1/60, scriptRefreshRate=1)