import copy

from bereshit.addons.PPO import Agent, ActorCritic
from bereshit import Object, Vector3, Core, Camera, BoxCollider, Rigidbody
from bereshit.addons.essentials import FPS_cam, CamController
from Names_types import Wall, Goal
from MoveToGoal import MoveToGoal


# cam = Object(position=Vector3(0, 0, -8)).add_component(Camera(), CamController(), FPS_cam())

def create_scene(y, shared_agent):

    floor = Object(position=Vector3(0, y-1, 0), size=Vector3(10, 1, 10)).add_component(BoxCollider(),
                                                                                      Rigidbody(isKinematic=True))
    wall1 = Object(position=Vector3(-5, y, 0), size=Vector3(1, 1, 10)).add_component(BoxCollider(),
                                                                                     Rigidbody(isKinematic=True),
                                                                                     Wall())
    wall2 = Object(position=Vector3(5, y, 0), size=Vector3(1, 1, 10)).add_component(BoxCollider(),
                                                                                    Rigidbody(isKinematic=True), Wall())
    wall3 = Object(position=Vector3(0, y, -5), size=Vector3(10, 1, 1)).add_component(BoxCollider(),
                                                                                     Rigidbody(isKinematic=True),
                                                                                     Wall())
    wall4 = Object(position=Vector3(0, y, 5), size=Vector3(10, 1, 1)).add_component(BoxCollider(),
                                                                                    Rigidbody(isKinematic=True), Wall())

    goal = Object(position=Vector3(2, y, 0)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Goal())

    agent = Object(position=Vector3(0, y,0)).add_component(BoxCollider(), Rigidbody(Freeze_Rotation=Vector3(1, 1, 1)), shared_agent,
                                   MoveToGoal(goal))
    main = Object(position=Vector3(0,y,0), size=Vector3(0,0,0), children=[floor, wall1, wall2, wall3, wall4, agent, goal])

    return main
scene = []
for i in range(5):
    shared_agent = Agent(
        obs_dim=6,
        action_dim_continuous=2,
        save_dir="data2",
        max_steps=50
    )
    scene += [create_scene(i * 4, copy.deepcopy(shared_agent))]

Core.run(scene, speed=10, Render=False)
