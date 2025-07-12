import math
import os
import random
import time
import torch
import CamController
import moveToGoal
import bereshit
import asyncio
import copy
import PPO
import threading

import playerController
import render





obj = bereshit.Object(position=(0, 0.0125, 0), size=(0.07, 0.07, 0.07),name="obj2")
obj.add_component("rigidbody", (bereshit.Rigidbody(mass=1,useGravity=True,isKinematic=False,friction_coefficient=1)))
obj.add_component("collider", bereshit.BoxCollider())

obj2 = bereshit.Object(position=(0, 1.0825, 0), size=(0.07, 0.07, 0.07),name="obj2")
obj2.add_component("rigidbody", (bereshit.Rigidbody(mass=1,useGravity=True,isKinematic=False,friction_coefficient=1)))
obj2.add_component("collider", bereshit.BoxCollider())

goal = bereshit.Object(position=(-5, 0.5, 5), size=(1, 1, 1),name="goal")
goal.add_component("rigidbody", (bereshit.Rigidbody(mass=5,useGravity=True,isKinematic=False,friction_coefficient=0.5,restitution=1)))
goal.add_component("collider", bereshit.BoxCollider())



wall = bereshit.Object(position=(0, 0.05, -1), size=(2, .3, 0.1),name="wall")
wall.add_component("collider", bereshit.BoxCollider())
wall2 = bereshit.Object(position=(0, 0.05, 1), size=(2, .3, .1),name="wall")
wall2.add_component("collider", bereshit.BoxCollider())
wall3 = bereshit.Object(position=(-1, 0.05, 0), size=(.1, .3, 2),name="wall")
wall3.add_component("collider", bereshit.BoxCollider())
wall4 = bereshit.Object(position=(0.1, 0.05, 0), size=(1, .3, 2),name="wall")
wall4.add_component("collider", bereshit.BoxCollider())

floor = bereshit.Object(position=(0, 0, 0), size=(2, 1, 2),name="floor", children=[])
floor.add_component("collider", bereshit.BoxCollider())
floor.add_component("rigidbody", (bereshit.Rigidbody(mass=999,isKinematic=True)))
wall_rigidbody = bereshit.Rigidbody(mass=999,isKinematic=True,restitution=0.1)
wall.add_component("rigidbody", wall_rigidbody)
wall2.add_component("", wall_rigidbody)
wall3.add_component("rigidbody", wall_rigidbody)
wall4.add_component("rigidbody", wall_rigidbody)
scene = bereshit.Object(position=(0, 0, 0), size=(0, 0, 0), children=[floor,obj,wall,wall2,wall3,obj2],name="scene") # floor,wall,wall2,wall3,wall4,,wall,wall2,wall3,wall4,
camera = bereshit.Object(position=(0, 0.4, -1), children=[],name="cam")
camera.add_component("camera", bereshit.camera())
camera.add_component('camController',CamController.CamController())

world = bereshit.Object(position=(0, 0, 0), size=(0, 0, 0), children=[scene,camera],name="world")

# scene.add_rotation(bereshit.Vector3(0,0,45),forall=True)

bereshit.dt = 1/120
startg = time.time()
FPS= 1
async def main_logic():
    start_wall_time = time.time()
    steps = 0
    speed = 5  # real time slip

    while True:
        steps += 1
        simulated_time = steps * bereshit.dt

        if steps % 10 == 0:
            world.update(bereshit.dt,chack=True)
        else:
            # Update simulation
            world.update(bereshit.dt)
        print(obj2.position)
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
