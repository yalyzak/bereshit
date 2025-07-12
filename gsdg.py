import math
import random

import bereshit
import time
import copy
import PPO
import types

import printG

rand = random.uniform(-9,9)
obj = bereshit.Object(position=(0, 0.5, 0), size=(1, 1, 1),name="obj")
obj.add_component("rigidbody", (bereshit.Rigidbody(mass=1,useGravity=True,isKinematic=False,friction_coefficient=0.5)))
obj.add_component("collider", bereshit.BoxCollider())
obj.collider.is_trigger = True


shared_model = PPO.ActorCritic(obs_dim=6, action_dim=4)
agent1 = PPO.Agent(obs_dim=6, action_dim=4, name="agent1")
shared_mode = agent1.model
shared_optimizer = agent1.optimizer

# obj.add_component("agent", agent1)

goal = bereshit.Object(position=(-5, 0.5, 0), size=(1, 1, 1),name="goal")
goal.add_component("rigidbody", (bereshit.Rigidbody(mass=5,useGravity=False,isKinematic=True,friction_coefficient=0,restitution=1)))
goal.add_component("collider", bereshit.BoxCollider())




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

scene1 = bereshit.Object(position=(0, 0, 0), size=(0, 0, 0), children=[floor,wall,wall2,wall3,wall4,obj,goal],name="scene")

def OnTriggerEnter(self, other):
    if other.obj.name == "wall":
        agent = self.obj.get_component("agent")
        agent.add_reward(-1, done=True)
        print(f"{agent.name} touched the wall")
        agent.end_episode()


    elif other.obj.name == "goal":
        agent = self.obj.get_component("agent")
        agent.add_reward(1, done=True)
        print(f"{agent.name} reached the goal")
        agent.end_episode()



obj.collider.OnTriggerEnter = types.MethodType(OnTriggerEnter, obj.collider)

scenes = []
objs = []
goals = []
def episode_begin(self):
    world.reset_to_default()
    # self.parent.reset_to_default()
for i in range(1):
    new_scene = copy.deepcopy(scene1)

    new_scene.set_position(bereshit.Vector3(new_scene.position.x,new_scene.position.y + i * 10,new_scene.position.z))
    # obj.collider.OnTriggerEnter() =
    obj = new_scene.search("obj")
    goal = new_scene.search("goal")

    agent = PPO.Agent(obs_dim=6, action_dim=4, name=f"Agent_{i}",optimizer=shared_optimizer,model=shared_model,max_steps=10)  # or use shared agent if needed
    agent.episode_begin = episode_begin

    obj.add_component("agent", agent)
    scenes.append(new_scene)
    objs.append(obj)
    goals.append(goal)
    new_scene.set_default_position()

world = bereshit.Object(position=(0, 0, 0), size=(0, 0, 0), children=[scene1],name="world")




import threading
import render
import time
import keyboard

def wasd_controller(obj):

    if keyboard.is_pressed('w'):
       obj.rigidbody.exert_force(bereshit.Vector3(0,0,100))

    if keyboard.is_pressed('a'):
        obj.rigidbody.exert_force(bereshit.Vector3(-100, 0, 0))

    if keyboard.is_pressed('s'):
        obj.rigidbody.exert_force(bereshit.Vector3(0, 0, -100))

    if keyboard.is_pressed('d'):
        obj.rigidbody.exert_force(bereshit.Vector3(100, 0, 0))

s = 0.06
def ai_controller(obj,action):
    if action == 0:
        obj.rigidbody.exert_force(bereshit.Vector3(0,0,1000))
    elif action == 1:
        obj.rigidbody.exert_force(bereshit.Vector3(-1000, 0,0))
    elif action == 2:
        obj.rigidbody.exert_force(bereshit.Vector3(0,0,-1000))
    elif action == 3:
        obj.rigidbody.exert_force(bereshit.Vector3(1000, 0, 0))





def update():
    while True:
        time.sleep(0.001)
        world.update(dt=s)
        # wasd_controller(obj)


def main_logic():
    agents = []
    objs_list = []
    goals_list = []

    # Extract agents, objs, and goals from each scene
    for scene in scenes:
        obj = scene.search("obj")
        goal = scene.search("goal")
        agent = obj.get_component("agent")

        objs_list.append(obj)
        goals_list.append(goal)
        agents.append(agent)

    while True:
        time.sleep(0.001)

        for i in range(len(scenes)):
            obj = objs_list[i]
            goal = goals_list[i]
            agent = agents[i]

            obs = [
                goal.local_position.x, goal.local_position.y, goal.local_position.z,
                obj.local_position.x, obj.local_position.y, obj.local_position.z
            ]
            action, _, _ = agent.get_action(obs)
            ai_controller(obj, action)



if __name__ == "__main__":

    # Start your logic thread
    # world.reset_to_default()
    logic_thread = threading.Thread(target=main_logic, daemon=True)
    logic_thread.start()
    # runer = threading.Thread(target=update, daemon=True)
    # runer.start()


    # Run the renderer in the main thread (required for Tkinter)
    render.run_renderer(world)