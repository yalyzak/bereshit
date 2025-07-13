import math
import os
import random
import time
import torch
import CamController
import moveToGoal
import bereshit
import asyncio
from bereshit import Vector3
import copy
import PPO
import threading

import playerController
import render


obj2 = bereshit.Object(
    position=(0, 0.15 + 0.2, 0),
    size=(0.1, 0.1, 0.1),
    name="obj2"
)
obj2.add_component("rigidbody", bereshit.Rigidbody(mass=1, useGravity=True, isKinematic=False))
obj2.add_component("collider", bereshit.BoxCollider())

obj = bereshit.Object(
    position=(0, 0.05 + 0.2, 0),
    size=(0.1, 0.1, 0.1),
    name="obj"
)
obj.add_component("rigidbody", bereshit.Rigidbody(mass=1, useGravity=True, isKinematic=False))
obj.add_component("collider", bereshit.BoxCollider())
# obj2.add_component("joint", bereshit.Joint(obj))

goal = bereshit.Object(
    position=(-0.5, 0.05, 0.5),
    size=(0.1, 0.1, 0.1),
    name="goal"
)
goal.add_component("rigidbody", bereshit.Rigidbody(mass=5, useGravity=True, isKinematic=False))
goal.add_component("collider", bereshit.BoxCollider())

wall = bereshit.Object(
    position=(0, 0.05, -1.0),
    size=(2.0, 0.1, 0.1),
    name="wall"
)
wall.add_component("collider", bereshit.BoxCollider())

wall2 = bereshit.Object(
    position=(0, 0.05, 1.0),
    size=(2.0, 0.1, 0.1),
    name="wall"
)
wall2.add_component("collider", bereshit.BoxCollider())

wall3 = bereshit.Object(
    position=(-1.0, 0.05, 0),
    size=(0.1, 0.1, 2.0),
    name="wall"
)
wall3.add_component("collider", bereshit.BoxCollider())

wall4 = bereshit.Object(
    position=(1.0, 0.05, 0),
    size=(0.1, 0.1, 2.0),
    name="wall"
)
wall4.add_component("collider", bereshit.BoxCollider())

floor = bereshit.Object(
    position=(0, -0.05, 0),
    size=(2.0, 0.1, 2.0),
    name="floor",
    children=[]
)
floor.add_component("collider", bereshit.BoxCollider())
floor.add_component("rigidbody", bereshit.Rigidbody(mass=999, isKinematic=True,useGravity=False))

wall_rigidbody = bereshit.Rigidbody(mass=1, isKinematic=True)
wall.add_component("rigidbody", wall_rigidbody)
wall2.add_component("rigidbody", wall_rigidbody)
wall3.add_component("rigidbody", wall_rigidbody)
wall4.add_component("rigidbody", wall_rigidbody)

scene = bereshit.Object(
    position=(0, 0, 0),
    size=(0, 0, 0),
    children=[floor,obj,wall,wall2,wall3
              ,wall4],
    name="scene"
)
camera = bereshit.Object(
    position=(0, 1.0, -0.6),
    children=[],
    name="cam"
)
camera.add_component("camera", bereshit.camera())
# camera.add_component("camController", CamController.CamController())
obj.add_component("playerController", playerController.PlayerController())


world = bereshit.Object(position=(0, 0, 0), size=(0, 0, 0), children=[scene,camera],name="world")
TARGET_FPS = 60
bereshit.dt = TARGET_FPS * 0.000165

dt = 1/600

startg = time.time()
FPS= 1
async def main_logic():
    start_wall_time = time.time()
    steps = 0
    speed = .1 # real time slip
    bereshit.dt = 1/((1/dt)/60) * speed
    scene.add_rotation(Vector3(0,0,45),forall=True)
    obj.add_rotation(Vector3(0,0,-45))

    while True:
        steps += 1
        simulated_time = steps * dt

        if steps % 10 == 0:
            world.update(dt,chack=True)
        else:
            # Update simulation
            world.update(dt)
        # Compute when, in wall clock time, this simulated time should happen
        # For double speed: simulated_time advances twice as fast as real time
        target_wall_time = start_wall_time + (simulated_time / speed)
        now = time.time()
        sleep_time = target_wall_time - now

        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

def start_async_loop():
    asyncio.run(main_logic())

if __name__ == "__main__":
    # Initialize world
    # world.reset_to_default()
    # start_async_loop()
    # Start async logic in a thread
    logic_thread = threading.Thread(target=start_async_loop, daemon=True)
    logic_thread.start()

    # Start rendering in main thread
    render.run_renderer(world)
