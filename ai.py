import math
import random

import bereshit
import time
import copy
import PPO

obj_ = bereshit.Object(local_position=(0, 0.5, 0), size=(1, 1, 1),name="obj")
obj_.add_component("rigidbody", (bereshit.Rigidbody(mass=1,useGravity=True,isKinematic=False,friction_coefficient=0.1)))
shared_agent = PPO.Agent(obs_dim=6, action_dim=4)


goal_ = bereshit.Object(local_position=(random.uniform(-9,9), 0.5, random.uniform(-9,9)), size=(1, 1, 1),name="goal")
goal_.add_component("rigidbody", (bereshit.Rigidbody(mass=5,useGravity=True,isKinematic=True)))


obj2 = bereshit.Object(local_position=(0, 2, 0), size=(1, 1, 1),name="obj2",children=[],weight=50)
obj2.add_component("rigidbody", (bereshit.Rigidbody(mass=5,useGravity=False)))

# wall = bereshit.Object(local_position=(0, 0.5, -10), size=(20, 1, 1),name="wall")
wall_ = bereshit.Object(local_position=(0, 0.5, -10), size=(20, 1, 1),name="wall")
wall2_ = bereshit.Object(local_position=(0, 0.5, 10), size=(20, 1, 1),name="wall")
wall3_ = bereshit.Object(local_position=(-10, 0.5, 0), size=(1, 1, 20),name="wall")
wall4_ = bereshit.Object(local_position=(10, 0.5, 0), size=(1, 1, 20),name="wall")


floor = bereshit.Object(local_position=(0, -0.5, 0), size=(20, 1, 20),name="floor", children=[])
wall_rigidbody = bereshit.Rigidbody(mass=999,isKinematic=True,restitution=0.1)
floor.add_component("rigidbody", (bereshit.Rigidbody(mass=999,isKinematic=True)))
wall_.add_component("rigidbody", wall_rigidbody)
wall2_.add_component("rigidbody", wall_rigidbody)
wall3_.add_component("rigidbody", wall_rigidbody)
wall4_.add_component("rigidbody", wall_rigidbody)


scene1 = bereshit.Object(local_position=(0, 0, 0), size=(0, 0, 0), children=[floor,wall_,wall2_,wall3_,wall4_,obj_,goal_],name="scene")

scenes = []
objs = []
goals = []
walls1 = []
walls2 = []
walls3 = []
walls4 = []

for i in range(1):
    new_scene = copy.deepcopy(scene1)
    new_scene.position.y += i * 10
    obj_i = new_scene.search("obj")
    goal_i = new_scene.search("goal")

    wall_i = new_scene.search("wall")
    wall2_i = new_scene.search("wall2")
    wall3_i = new_scene.search("wall3")
    wall4_i = new_scene.search("wall4")

    scenes.append(new_scene)
    objs.append(obj_i)
    walls1.append(wall_i)
    walls2.append(wall2_i)
    walls3.append(wall3_i)
    walls4.append(wall4_i)
    goals.append(goal_i)
for agent_obj in objs:
    agent_obj.add_component("agent", shared_agent)
world = bereshit.Object(local_position=(0, 0, 0), size=(0, 0, 0), children=scenes,name="world")


def ai_controller(obj,action):
    if action == 0:
        obj.rigidbody.velocity.z += 0.5  # w
    elif action == 1:
        obj.rigidbody.velocity.x += -0.5  # a
    elif action == 2:
        obj.rigidbody.velocity.z += -0.5  # s
    elif action == 3:
        obj.rigidbody.velocity.x += 0.5  # d

import PPO
import threading
import render
import torch
import os


def main_logic():
    agent = shared_agent  # one global agent shared across all objs

    # === Load Model ===
    os.makedirs("checkpoints", exist_ok=True)
    if os.path.exists("checkpoints/ppo_agent.pt"):
        print("Load saved model? (y/n)")
        if input().lower() == "y":
            agent.model.load_state_dict(torch.load("checkpoints/ppo_agent.pt"))
            print("✅ Loaded existing PPO model")
        else:
            print("🚀 Starting with a new PPO model")
    else:
        print("🚀 Starting with a new PPO model")

    # === Constants ===
    MAX_EPISODES = 5000
    MAX_STEPS = 100
    UPDATE_INTERVAL = 5

    for episode in range(MAX_EPISODES):
        # Reset environment
        for obj in objs:
            obj.reset_to_default()
        for goal in goals:
            goal.reset_to_default()

        done_flags = {obj: False for obj in objs}

        for step in range(MAX_STEPS):
            time.sleep(0.01)
            world.update(dt=0.01)

            for obj, goal, wall1, wall2, wall3, wall4 in zip(objs, goals, walls1, walls2, walls3, walls4):
                if done_flags[obj]:
                    continue  # skip agents that are done

                agent = obj.get_component("agent")

                obs = [
                    goal.position.x, goal.position.y, goal.position.z,
                    obj.position.x, obj.position.y, obj.position.z
                ]

                action, logp, val = agent.get_action(obs)
                ai_controller(obj, action)

                # Collisions
                hit_goal = obj.collider.check_collision(goal) is not None
                hit_wall = any(obj.collider.check_collision(w) is not None for w in [wall1, wall2, wall3, wall4])

                reward = 0.0
                done = False

                if hit_wall:
                    reward = -1.0
                    done = True
                    print("bad")

                elif hit_goal:
                    print("good")
                    reward = 1.0
                    done = True

                if done:
                    done_flags[obj] = True
                    obj.reset_to_default()
                    goal.reset_to_default()

                next_obs = [
                    goal.position.x, goal.position.y, goal.position.z,
                    obj.position.x, obj.position.y, obj.position.z
                ]

                agent.store((obs, action, logp, reward, val, done))

            if all(done_flags.values()):
                break  # End episode early if all agents are done

        # === PPO Update ===
        if episode % UPDATE_INTERVAL == 0:
            agent.update()
            torch.save(agent.model.state_dict(), "checkpoints/ppo_agent.pt")
            print(f"Episode {episode} ✅ model updated and saved")

if __name__ == "__main__":

    # Start your logic thread
    world.reset_to_default()

    logic_thread = threading.Thread(target=main_logic, daemon=True)
    logic_thread.start()

    # Run the renderer in the main thread (required for Tkinter)
    render.run_renderer(world)