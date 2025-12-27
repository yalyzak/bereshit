import random
import torch

from bereshit import World, Vector3

episode = 0
UPDATE_INTERVAL = 1000

class moveToGoal:
    def __init__(self,goal):
        self.goal = goal


    def OnTriggerEnter(self, other):
        if other.obj.name == "wall":
            agent = self.parent.get_component("Agent")
            agent.add_reward(-1, done=True)
            # print(f"{agent.name} touched the wall")
        elif other.obj.name == "goal":
            agent = self.parent.get_component("Agent")
            agent.add_reward(10, done=True)
            # print(f"{agent.name} reached the goal")

    def On_episode_begin(self):
        # world.reset_to_default()
        # self.parent.parent.reset_to_default()
        self.parent.local_position = Vector3(random.uniform(-5, 5), 1, random.uniform(-5, 5))
        # self.parent.local_position = Vector3(5, 1, 7)

        self.parent.Rigidbody.velocity = Vector3(0,0,0)
        # self.parent.movetogoal.goal.local_position = Vector3(random.uniform(-5, 5), 0.5,
        #                                                                random.uniform(-5, 5))
    def apply_continuous_action(self, action):
        speed = 100
        # action is a numpy array like [fx, fy, fz]
        velocity = Vector3(action[0], 0, action[1]) * speed * World.tick
        self.parent.Rigidbody.velocity += velocity
    def Update(self,dt):
        global episode, UPDATE_INTERVAL
        episode += 1
        # === Observation ===
        agent_comp = self.parent.get_component("Agent")
        if agent_comp.ready_to_update:
            agent_comp.ready_to_update = False
            agent_comp.end_episode()
        goal_pos = self.goal.local_position
        obj_pos = self.parent.local_position

        obs = [
            goal_pos.x, goal_pos.y, goal_pos.z,
            obj_pos.x, obj_pos.y, obj_pos.z,
        ]

        # === Get Action ===
        action, _, _ = agent_comp.get_continuous_action(obs)

        # === Apply Action ===
        self.apply_continuous_action(action)

        # Optional: tiny time reward to encourage faster episodes
        agent_comp.add_reward(-0.001, done=False)
        if episode % UPDATE_INTERVAL == 0:
            torch.save(agent_comp.model.state_dict(), "checkpoints/ppo_agent.pt")
            print(f"Episode {episode} ✅ model updated and saved")