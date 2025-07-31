import Core
from bereshit import Object,Camera,Vector3, Rigidbody, BoxCollider, Mesh_rander, FixJoint,Quaternion
from FPS_cam import rotate
from CamController import CamController
from debug import debug



obj_small = Object(position=(0, 2.0,    0), size=(1, 1.5, 1),rotation=(0,0,50),name='obj_small')
joint = Object(position=(0, 5,    0), size=(1, 1, 1),name='joint')
floor = Object(position=(0,-1,0),size=(10,1,10),rotation=(0,0,0),name='floor')





obj_small.add_component(Rigidbody(useGravity=True,velocity=Vector3(0,-1,0)))
obj_small.add_component(BoxCollider())



camera = Object(size=(1,1,1),position=(0,0,-5),name='camera')

camera.add_component(Camera())
camera.add_component(Mesh_rander(shape="empty"))

camera.add_component(rotate())
camera.add_component(CamController())
camera.add_component(Rigidbody(isKinematic=True))
camera.add_component(BoxCollider())
camera.add_component(BoxCollider())

joint.add_component(Rigidbody(isKinematic=True))
joint.add_component(BoxCollider())

# joint.add_component(FixJoint(other_object=obj_small))
joint.add_component(debug())
# obj_small.add_component("d", debug())

floor.add_component(Rigidbody(isKinematic=True))
floor.add_component(BoxCollider())

scene = Object(
    position=Vector3(0, 0, 0),
    size=(0, 0, 0),
    children=[obj_small,floor],
    name="scene"
)
world = Object(position=(0, 0, 0), size=(0, 0, 0), children=[camera,scene],name='world')

Core.run(world)