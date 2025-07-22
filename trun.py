import bereshit
from bereshit import Camera
from bereshit import Object
from bereshit import Vector3
from bereshit import BoxCollider
from bereshit import Rigidbody
from bereshit import Mesh_rander
from bereshit import MeshCollider
from bereshit import SphereCollider
import FPS_cam
from CamController import CamController
from playerController import PlayerController
import Core

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

floor_length = 10
floor_width = 10
# --- Camera ---
camera = Object(
    size=(0,0,0),
    position=Vector3(0, 2.0, -floor_length / 2 + 2.0),
    name="camera"
)
camera.add_component("camera", Camera(shading="wire"))
camera.add_component('camController',CamController(speed=0.02))
camera.add_component("asd", FPS_cam.rotate())
# camera.add_component("mesh", Mesh_rander(shape='box'))


floor = Object(
    position=Vector3(0, -0.05, 0),
    size=(floor_width, 0.1, floor_length),
    name="floor"
)
floor.add_component("collider", BoxCollider())
floor.material.kind = "Asphalt"
floor.add_component("rigidbody", Rigidbody(mass=999, isKinematic=True, useGravity=False))

box = Object(position=(0,1,0),rotation=(90,0,0))
box.add_component("rigidbody", Rigidbody(useGravity=False))
box.add_component("collider", BoxCollider())
box.add_component("mesh", Mesh_rander(obj_path="models/Cup.obj"))
# box.add_component("asd", Rotate.rotate())
# box.add_component("camera", Camera())

# cup = Object(size=(1,1,1),position=(0,10,0),name="cup")
# cup.add_component("rigidbody", Rigidbody())
# cup.add_component("mesh", Mesh_rander(obj_path="models/Cup.obj"))
# cup.add_component("collider", BoxCollider())
# cup.add_component("material", bereshit.Material(color=(0, 0, 255)))

ball = Object(position=(0,2,0))
ball.add_component("rigidbody", Rigidbody(restitution=1))
ball.add_component("mesh", Mesh_rander(obj_path="models/Cup.obj"))
ball.add_component("collider", BoxCollider())



arm = Object(position=(0, 0, -5.5/2), rotation=(0, 0, 0), size=(0.1, 0.1, 0.4))

servo = Object(position=(0, 0, 0), rotation=(0, 0, 0), size=(2, 6, 5.5))


# servo.add_component("mesh", Mesh_rander(obj_path="models/servo.obj"))



scene = Object(
    position=Vector3(0, 0, 0),
    size=(0, 0, 0),
    children=[camera,floor,servo,arm],
    name="scene"
)

world = Object(
    position=Vector3(0, 0, 0),
    size=(0, 0, 0),
    children=[scene],
    name="world"
)

Core.run(world)