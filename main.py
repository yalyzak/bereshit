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
import render

obj = bereshit.Object(position=(0, 0.5, 0), size=(1, 1, 1),name="obj")
obj.add_component("rigidbody", (bereshit.Rigidbody(mass=1,useGravity=True,isKinematic=False,friction_coefficient=0.5)))
obj.add_component("collider", bereshit.BoxCollider())
obj.collider.is_trigger = True


shared_model = PPO.ActorCritic(obs_dim=6, action_dim_continuous=2)
shared_optimizer = torch.optim.Adam(shared_model.parameters(), lr=2.5e-4)

agent = PPO.Agent(obs_dim=6, action_dim_continuous=2, name="agent",model=shared_model,optimizer=shared_optimizer)

obj.add_component("agent", agent)

goal = bereshit.Object(position=(-5, 0.5, 5), size=(1, 1, 1),name="goal")
goal.add_component("rigidbody", (bereshit.Rigidbody(mass=5,useGravity=True,isKinematic=False,friction_coefficient=0.5,restitution=1)))
goal.add_component("collider", bereshit.BoxCollider())
obj.add_component('movetogoal',moveToGoal.moveToGoal(goal))
# goal.add_component('player_controller',playerController.PlayerController())



wall = bereshit.Object(position=(0, 0.5, -10), size=(20, 1, 1),name="wall")
wall.add_component("collider", bereshit.BoxCollider())
wall2 = bereshit.Object(position=(0, 0.5, 10), size=(20, 1, 1),name="wall")
wall2.add_component("collider", bereshit.BoxCollider())
wall3 = bereshit.Object(position=(-10, 0.5, 0), size=(1, 1, 20),name="wall")
wall3.add_component("collider", bereshit.BoxCollider())
wall4 = bereshit.Object(position=(10, 0.5, 0), size=(1, 1, 20),name="wall")
wall4.add_component("collider", bereshit.BoxCollider())

floor = bereshit.Object(position=(0, -0.5, 0), size=(20, 1, 20),name="floor", children=[])
floor.add_component("collider", bereshit.BoxCollider())
floor.add_component("rigidbody", (bereshit.Rigidbody(mass=999,isKinematic=True)))
wall_rigidbody = bereshit.Rigidbody(mass=999,isKinematic=True,restitution=0.1)
wall.add_component("rigidbody", wall_rigidbody)
wall2.add_component("rigidbody", wall_rigidbody)
wall3.add_component("rigidbody", wall_rigidbody)
wall4.add_component("rigidbody", wall_rigidbody)


if os.path.exists("checkpoints/ppo_agent.pt"):
    print("Load saved model? (y/n)")
    if input().lower() == "y":
        agent.model.load_state_dict(torch.load("checkpoints/ppo_agent.pt"))
        print("✅ Loaded existing PPO model")
    else:
        print("🚀 Starting with a new PPO model")
else:
    print("🚀 Starting with a new PPO model")

scene = bereshit.Object(position=(0, 0, 0), size=(0, 0, 0), children=[floor,wall,wall2,wall3,wall4,obj,goal],name="scene")

scenes = []
objs = []
goals = []
for i in range(15):
    new_scene = copy.deepcopy(scene)

    new_scene.set_position(bereshit.Vector3D(new_scene.position.x,new_scene.position.y + i * 10,new_scene.position.z))

    obj = new_scene.search("obj")
    goal = new_scene.search("goal")

    scenes.append(new_scene)
    objs.append(obj)
    goals.append(goal)
    new_scene.set_default_position()

    # obj.agent.parent = obj
    obj.agent.name += f"_{i}"
    obj.movetogoal.goal = goal

camera = bereshit.Object(
        position=(0, 10, -6),
        rotation=(0, 0, 0),
        size=(0, 0, 0),
        children=[]
    )
camera.add_component("camera", bereshit.camera())
camera.add_component('camController',CamController.CamController())
scenes.append(camera)
world = bereshit.Object(position=(0, 0, 0), size=(0, 0, 0), children=scenes,name="world")


TARGET_FPS = 60 * 2.5
bereshit.dt = TARGET_FPS * 0.000165

async def main_logic():
    while True:
        frame_start = time.perf_counter()

        world.update(dt=bereshit.dt)
        # Measure how long update took
        elapsed = time.perf_counter() - frame_start
        # Compute remaining time to sleep
        sleep_time = max(0, bereshit.dt - elapsed)

        await asyncio.sleep(sleep_time)

def start_async_loop():
    asyncio.run(main_logic())

if __name__ == "__main__":
    # Initialize world
    world.reset_to_default()
    # start_async_loop()
    # Start async logic in a thread
    logic_thread = threading.Thread(target=start_async_loop, daemon=True)
    logic_thread.start()

    # Start rendering in main thread
    render.run_renderer(world)