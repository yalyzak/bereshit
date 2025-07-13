import copy
import random

import Core
import Goal
import bereshit
from bereshit import Vector3
import playerController
import CamController

# --- Floor ---
floor_length = 20.0
floor_width = 20.0

floor = bereshit.Object(
    position=Vector3(0, -0.05, 0),
    size=(floor_width, 0.1, floor_length),
    name="floor"
)
floor.add_component("collider", bereshit.BoxCollider())
floor.material.kind = "Asphalt"
floor.add_component("rigidbody", bereshit.Rigidbody(mass=999, isKinematic=True, useGravity=False))


# --- Camera ---
camera = bereshit.Object(
    position=Vector3(0, 2.0, -floor_length / 2 + 2.0),
    name="camera"
)
camera.add_component("camera", bereshit.camera())

obj2 = bereshit.Object(
    position=Vector3(0, 0.3, -floor_length / 2 + 3.0),
    size=(0.4, 0.2, 0.6),
    name="obj2",
    children=[]
)
obj2.material.kind = "Asphalt"
obj2.add_component("collider", bereshit.BoxCollider())
obj2.add_component("rigidbody", bereshit.Rigidbody(mass=1, useGravity=True, isKinematic=False))
# obj2.add_component("playerController", playerController.PlayerController())
# obj2.rigidbody.velocity += Vector3(1,0,0)

obj = bereshit.Object(
    position=Vector3(0, 0.3, -floor_length / 2 + 3.0),
    size=(0.4, 0.2, 0.6),
    name="obj",
    children=[camera]
)


obj.add_component("collider", bereshit.BoxCollider())
obj.add_component("rigidbody", bereshit.Rigidbody(mass=1, useGravity=True, isKinematic=False))
obj.rigidbody.velocity += Vector3(1,0,0)
# obj.add_component("joint",bereshit.FixJoint(obj2))
obj.add_component("playerController", playerController.PlayerController())

# --- Scene ---
scene = bereshit.Object(
    position=Vector3(0, 0, 0),
    size=(0, 0, 0),
    children=[floor, obj,obj2],
    name="scene"
)

world = bereshit.Object(
    position=Vector3(0, 0, 0),
    size=(0, 0, 0),
    children=[scene],
    name="world"
)

Core.run(world)
