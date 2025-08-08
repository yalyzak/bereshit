import Core
from bereshit import Object,Camera,Vector3, Rigidbody, BoxCollider, Mesh_rander, FixJoint,Quaternion
from FPS_cam import rotate
from CamController import CamController
from debug import debug


obj_small = Object(position=(0, 0,    0), size=(1, 1, 1),rotation=(0,0,10),name='obj_small')
obj_big = Object(position=(0, 3,    0), size=(1, 1, 1),rotation=(0,0,0),name='obj_big')

joint = Object(position=(0, 5,    0), size=(1, 1, 1),name='joint')
floor = Object(position=(0,-1,0),size=(100,1,100),rotation=(0,0,0),name='floor')
eamty = Object(size=(5,1,1),name="op",position=(5,0,0))




obj_small.add_component(Rigidbody(useGravity=True,velocity=Vector3(0,0,0)))
obj_small.add_component(BoxCollider())

obj_big.add_component(Rigidbody(useGravity=True,velocity=Vector3(0,-4,0)))
obj_big.add_component(BoxCollider())

camera = Object(size=(1,1,1),position=(0,0,-5),name='camera')

camera.add_component(Camera())
camera.add_component(Mesh_rander(shape="empty"))

camera.add_component(rotate())
camera.add_component(CamController())
camera.add_component(Rigidbody(isKinematic=True))
camera.add_component(BoxCollider())

joint.add_component(Rigidbody(isKinematic=True))
joint.add_component(BoxCollider())

# joint.add_component(FixJoint(other_object=obj_small))
# obj_small.add_component(debug())
# eamty.add_component(debug())

floor.add_component(Rigidbody(isKinematic=True,velocity=Vector3(0,0,0)))
floor.add_component(BoxCollider())

scene = Object(
    position=Vector3(0, 0, 0),
    size=(0, 0, 0),
    children=[obj_small,floor,camera,obj_big],
    # children=[obj_big,obj_small, floor, camera],

    # children=[floor, obj_small,camera],
    # children=[obj_small,floor, camera],

    name="scene"
)


Core.run(scene,gizmos=True)