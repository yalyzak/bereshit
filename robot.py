import Core
from bereshit import Object, Camera, Vector3, Rigidbody, BoxCollider, FixJoint
from FPS_cam import rotate
from CamController import CamController
servo_right_down = Object(position=(0, -6.25, 0), rotation=(0, 0, 0), size=(6, 6.5, 2))

servo_right_down_side = Object(position=(0, -10.75, 0), rotation=(0, 0, 0), size=(6, 6.5, 2), children=[servo_right_down])

lower_leg_r = Object(size=(3.8, 15, 3.8), position=(0, -10.75, 0), children=[servo_right_down_side])

servo_right = Object(position=(0, -10.75, 0), rotation=(0, 0, 0), size=(6, 6.5, 2), children=[lower_leg_r])

upper_leg_r = Object(size=(3.8, 15, 3.8), position=(0, -10.75, 0), children=[servo_right])

servo_right_up_side = Object(position=(0, -6.25, 0), rotation=(0, 0, 0), size=(2, 6.5, 6), children=[upper_leg_r])

spine_side_1 = Object(position=(0, -6.25, 0), rotation=(0, 0, 0), size=(2, 6.5, 6), children=[])

spine_1 = Object(position=(0, -6.25, 0), rotation=(0, 0, 0), size=(2, 6.5, 6), children=[spine_side_1])

servo_right_up = Object(position=(7.75, -3.25, -1.5), rotation=(0, 0, 0), size=(6, 6.5, 2), children=[servo_right_up_side])

leg_right = Object(size=(0, 0, 0), children=[servo_right_up])

# left leg
servo_left_down = Object(position=(0, -6.25, 0), rotation=(0, 0, 0), size=(6, 6.5, 2))

servo_left_down_side = Object(position=(0, -10.75, 0), rotation=(0, 0, 0), size=(6, 6.5, 2), children=[servo_left_down])

lower_leg_l = Object(size=(3.8, 15, 3.8), position=(0, -10.75, 0), children=[servo_left_down_side])

servo_left = Object(position=(0, -10.75, 0), rotation=(0, 0, 0), size=(6, 6.5, 2), children=[lower_leg_l])

upper_leg_l = Object(size=(3.8, 15, 3.8), position=(0, -10.75, 0), children=[servo_left])

servo_left_up_side = Object(position=(0, -6.25, 0), rotation=(0, 0, 0), size=(4, 3.9, 2), children=[])

leg_left2 = Object(size=(4, 4, 2), position=(-14.5, -3.5/2 + -4, 0), children=[])

leg_left2.add_component("rigidbody", Rigidbody(useGravity=True))
leg_left2.add_component("collider", BoxCollider())

leg_left = Object(size=(5.2, 3.5, 4), position=(-14.5, -3.5/2, 0), children=[leg_left2])

leg_left.add_component("rigidbody", Rigidbody(useGravity=False))
leg_left.add_component("collider", BoxCollider())
# leg_left.add_component("joint", FixJoint(other_object=leg_left2))

hip = Object(position=(0, 3.5/2, 6), size=(38.2, 3.5, 3.8), children=[])  # previously also leg_right and spine_1

hip.add_component("rigidbody", Rigidbody(useGravity=True))
hip.add_component("collider", BoxCollider())

hip.add_component("joint", FixJoint(other_object=leg_left))

ground = Object(position=(0, -64, 0), size=(20, 1, 20))
ground.add_component("rigidbody", Rigidbody(isKinematic=True))
ground.add_component("collider", BoxCollider())

camera = Object(size=(1,1,1))

camera.add_component("camera",Camera())

camera.add_component("FPS_cam",rotate())
camera.add_component("CamController",CamController())
camera.add_component("rigidbody", Rigidbody(useGravity=True))
camera.add_component("collider", BoxCollider())

scene = Object(
    position=Vector3(0, 0, 0),
    size=(0, 0, 0),
    children=[hip,ground],
    name="scene"
)
world = Object(position=(0, 0, 0), size=(0, 0, 0), children=[camera,scene])

Core.run(world)