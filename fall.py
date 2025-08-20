import copy

from bereshit import Object,Camera,Vector3, Rigidbody, BoxCollider, Mesh_rander, FixJoint,Quaternion
from FPS_cam import rotate
from CamController import CamController
from debug import debug
import Core


obj_small = Object(position=(0,20,    0), size=(1, 1, 1),rotation=(0,0,0),name='obj_small')
obj_big = Object(position=(0,19,    0), size=(1, 1, 1),rotation=(0,0,0),name='obj_big')
obj_big2 = Object(position=(0,3,    0), size=(1, 1, 1),rotation=(0,0,0),name='obj_big')

obj = Object(position=(0, 5,    0), size=(1, 1, 1),rotation=(0,0,0),name='obj_big')

joint = Object(position=(0, 5,    0), size=(1, 1, 1),name='joint')
floor = Object(position=(0,0,0),size=(100,1,100),rotation=(0,0,0),name='floor')
eamty = Object(size=(5,1,1),name="op",position=(5,0,0))


r = 0.1

obj_small.add_component(Rigidbody(useGravity=False,velocity=Vector3(0,-1,0),restitution=r))
obj_small.add_component(BoxCollider())

obj_big.add_component(Rigidbody(useGravity=False,velocity=Vector3(0,0,0),restitution=r))
obj_big.add_component(BoxCollider())
obj_big2.add_component(Rigidbody(useGravity=True,velocity=Vector3(0,0,0),restitution=r))
obj_big2.add_component(BoxCollider())
camera = Object(size=(1,1,1),position=(0,0,-5),name='camera')

camera.add_component(Camera())
camera.add_component(Mesh_rander(shape="empty"))

# camera.add_component(rotate())
camera.add_component(CamController())
camera.add_component(Rigidbody(isKinematic=True))
camera.add_component(BoxCollider())

joint.add_component(Rigidbody(isKinematic=True))
joint.add_component(BoxCollider())

# joint.add_component(FixJoint(other_object=obj_small))
obj_small.add_component(debug(obj_big))
# eamty.add_component(debug())
g = copy.deepcopy(obj_small)
g.position.y += 3
floor.add_component(Rigidbody(isKinematic=True,velocity=Vector3(0,0,0),restitution=r))
floor.add_component(BoxCollider())
g2 = copy.deepcopy(obj_small)
g2.position.y += 1

scene = Object(
    position=Vector3(0, 0, 0),
    size=(0, 0, 0),
    # children=[obj_small,floor,camera,obj_big,obj_big2],
    children=[obj_big,obj_small, camera],
    #
    # children=[floor, obj_small,camera],
    # children=[obj_small,floor, camera],

    name="scene"
)


Core.run(scene,gizmos=False,speed=1,tick=1/60,scriptRefreshRate=1)