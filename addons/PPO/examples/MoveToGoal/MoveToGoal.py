from bereshit.addons.PPO import Agent, ActorCritic
from bereshit import Object, Vector3, Core, Camera, BoxCollider, Rigidbody
from bereshit.addons.essentials import FPS_cam, CamController
import random


class MoveToGoal:
    def __init__(self, goal):
        self.Agent = None
        self.goal = goal
        self.speed = 500

    def attach(self, parent):
        self.Agent = parent.get_component("Agent")

    def OnEpisodeBegin(self):
        self.parent.reset_to_default()
        self.goal.reset_to_default()
        self.parent.local_position += Vector3(random.uniform(-4, 4), 0, random.uniform(-4, 4))
        self.goal.local_position += Vector3(random.uniform(-4, 4), 0, random.uniform(-4, 4))

    def Update(self, dt):
        pos = self.parent.local_position
        pos2 = self.goal.local_position
        obs = [pos.x, pos.y, pos.z, pos2.x, pos2.y, pos2.z]
        action, _, _ = self.Agent.get_continuous_action(obs)
        self.Move(action[0], action[1], dt)
        self.addRewardByDistance(pos, pos2)
    def Move(self, x, z, dt):
        self.parent.Rigidbody.velocity += Vector3(x, 0, z) * dt * self.speed

    def addRewardByDistance(self, pos, pos2):
        distance = (pos - pos2).magnitude()
        reward = -distance * 0.01
        self.Agent.add_reward(reward)

    def OnCollisionEnter(self, Collision):
        if Collision.other.parent.get_component("Wall"):
            self.Agent.add_reward(-1)
            self.Agent.end_episode()
        elif Collision.other.parent.get_component("Goal"):
            self.Agent.add_reward(1)
            self.Agent.end_episode()
