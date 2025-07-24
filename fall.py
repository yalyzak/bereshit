import Core
from bereshit import Object,Camera,Vector3, Rigidbody, BoxCollider, Mesh_rander
from FPS_cam import rotate
from CamController import CamController
from debug import debug


obj_big = Object(position=(0, -10, 0), size=(1, 2, 1))
obj_small = Object(position=(0, 1.0,    0), size=(1, 1, 1))


obj_small.add_component("rigidbody", Rigidbody(useGravity=True))
obj_small.add_component("collider", BoxCollider())
obj_small.add_component("d", debug())

obj_big.add_component("rigidbody", Rigidbody(isKinematic=True))
obj_big.add_component("mesh", Mesh_rander(obj_path="models/gli nahami.obj"))
obj_big.add_component("collider", BoxCollider())

camera = Object(size=(0,0,0))

camera.add_component("camera",Camera())

camera.add_component("FPS_cam",rotate())
camera.add_component("CamController",CamController())
camera.add_component("rigidbody", Rigidbody(useGravity=True))
camera.add_component("collider", BoxCollider())
scene = Object(
    position=Vector3(0, 0, 0),
    size=(0, 0, 0),
    children=[obj_big,obj_small],
    name="scene"
)
world = Object(position=(0, 0, 0), size=(0, 0, 0), children=[camera,scene])

Core.run(world)