"""Heavy + Light scene -- run with the full Bereshit renderer via Core.run().

Scene:
  - 1 camera (looking at the action)
  - 1 heavy block (mass=50, stationary)
  - 1 light ball  (mass=1,  initial velocity=(5, 3, 0))
  - light is hinged to heavy around the Y axis
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from bereshit import Object, Vector3, Camera, Rigidbody, BoxCollider
from bereshit.HingeJoint import HingeJoint
from bereshit.Core import run
from bereshit.addons.essentials import CamController, FPS_cam

# --- Camera ---
cam = Object(position=(0, 3, -12), name="camera")
cam.add_component([Camera(shading="solid"), CamController(), FPS_cam(), BoxCollider()])

# --- Heavy body (kinematic = fixed in space, acts as anchor) ---
heavy = Object(position=(0, 5, 0), size=(1, 1, 1), name="heavy_block")
heavy.add_component(Rigidbody(mass=1.0, isKinematic=False, useGravity=False))
heavy.add_component(BoxCollider())

# --- Light body (swings like a pendulum) ---
light = Object(position=(0, 2, 0), size=(0.5, 0.5, 0.5), name="light_ball")
light.add_component(Rigidbody(
    mass=1.0,
    velocity=Vector3(5, 0, 0),
    useGravity=False,
))
light.add_component(BoxCollider())

# --- Hinge joint: heavy -> light, pivot at heavy's position ---
hinge = HingeJoint(body_b=light, axis=Vector3(0, 0, 1), friction_coefficient=0.05,
                   anchor=Vector3(0, 5, 0))  # pivot at heavy's center
heavy.add_component(hinge)

# --- Floor for visual reference ---
floor = Object(position=(0, -2, 0), size=(20, 0.2, 20), name="floor")
floor.add_component(Rigidbody(mass=1.0, isKinematic=True, useGravity=False))
floor.add_component(BoxCollider())

# --- Run ---
run(
    scene=[cam, heavy, light, floor],
    gravity=Vector3(0, -9.8, 0),
    speed=1,
    tick=1/60,
)
#