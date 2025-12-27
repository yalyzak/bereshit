import torch

from bereshit import Rigidbody, BoxCollider, Object, Core, Camera, Vector3
from bereshit.addons.PPO.PPO import Agent,ActorCritic
from bereshit.addons.essentials import CamController, FPS_cam
from PPO_scene.ai import moveToGoal
shared_model = ActorCritic(obs_dim=6, action_dim_continuous=2)
shared_optimizer = torch.optim.Adam(shared_model.parameters(), lr=2.5e-4)

agent = Agent(obs_dim=6, action_dim_continuous=2, name="agent",model=shared_model,optimizer=shared_optimizer)
agent.model.load_state_dict(torch.load("checkpoints/ppo_agent.pt"))

Goal = Object(name="goal",position=Vector3(1,1,1)).add_component([BoxCollider(),Rigidbody(isKinematic=True)])
obj = Object(position=(0, 2, 0), size=(1, 1, 1),name="obj")
obj.add_component([Rigidbody(mass=1,useGravity=True,friction_coefficient=0.5), BoxCollider(is_trigger=True),agent,moveToGoal(Goal)])

floor = Object(position=Vector3(0,0,0),size=(20,1,20),rotation=Vector3(0,0,0)).add_component([BoxCollider(),Rigidbody(isKinematic=True)])
wall = Object(position=(0, 0.5, -10), size=(20, 1, 1),name="wall")
wall.add_component([BoxCollider(), Rigidbody(isKinematic=True)])
wall2 = Object(position=(0, 0.5, 10), size=(20, 1, 1),name="wall")
wall2.add_component([BoxCollider(), Rigidbody(isKinematic=True)])
wall3 = Object(position=(-10, 0.5, 0), size=(1, 1, 20),name="wall")
wall3.add_component([BoxCollider(), Rigidbody(isKinematic=True)])
wall4 = Object(position=(10, 0.5, 0), size=(1, 1, 20),name="wall")
wall4.add_component([BoxCollider(), Rigidbody(isKinematic=True)])

camera = Object(position=Vector3(0,5,-2)).add_component([Camera(),CamController(),BoxCollider(),Rigidbody(isKinematic=True)])

Core.run([obj, camera, Goal, floor,wall,wall2,wall3,wall4],speed=10)