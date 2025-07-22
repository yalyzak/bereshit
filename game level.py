import copy
import random

import Core
import Goal
import bereshit
from bereshit import Vector3
import playerController
import CamController

# --- Floor ---
floor_length = 50.0
floor_width = 4.0

floor = bereshit.Object(
    position=Vector3(0, -0.05, 0),
    size=(floor_width, 0.1, floor_length),
    name="floor"
)
floor.add_component("collider", bereshit.BoxCollider())
floor.material.kind = "Asphalt"
floor.add_component("rigidbody", bereshit.Rigidbody(mass=999, isKinematic=True, useGravity=False))

# --- Walls (Track Boundaries) ---
walls = []
wall_positions = [
    (0, 0.5, -floor_length / 2),    # back
    (0, 0.5, floor_length / 2),     # front
    (-floor_width / 2, 0.5, 0),     # left
    (floor_width / 2, 0.5, 0)       # right
]
for i, pos in enumerate(wall_positions):
    wall = bereshit.Object(
        position=Vector3(*pos),
        size=(
            floor_width if i < 2 else 0.1,
            1.0,
            0.1 if i < 2 else floor_length
        ),
        name=f"wall{i}"
    )
    wall.add_component("collider", bereshit.BoxCollider())
    wall.material.kind = "Concrete"
    wall.add_component("rigidbody", bereshit.Rigidbody(mass=999, isKinematic=True))
    walls.append(wall)

# --- Random Obstacles as track barriers ---
obstacles = []
prototype_obstacle = bereshit.Object(
    position=Vector3(0, 0, 0),
    size=(0.4, 0.3, 1.0),
    name="obstacle"
)
prototype_obstacle.add_component("collider", bereshit.BoxCollider())
prototype_obstacle.material.kind = "Barrier"
prototype_obstacle.material.color = "red"
prototype_obstacle.add_component("rigidbody", bereshit.Rigidbody(mass=5, useGravity=False, isKinematic=True))

num_obstacles = 15
for i in range(num_obstacles):
    # Random x in track width minus margin
    x = random.uniform(-floor_width / 2 + 0.5, floor_width / 2 - 0.5)
    # Random z along the length
    z = random.uniform(-floor_length / 2 + 2.0, floor_length / 2 - 2.0)
    obstacle = copy.deepcopy(prototype_obstacle)
    obstacle.name = f"barrier{i}"
    obstacle.position = Vector3(x, 0.05, z)
    obstacles.append(obstacle)

# --- Camera ---
camera = bereshit.Object(
    position=Vector3(0, 2.0, -floor_length / 2 + 2.0),
    name="camera"
)
camera.add_component("camera", bereshit.Camera())
player = bereshit.Object(
    position=Vector3(0, 0.5, -floor_length / 2 + 3.0),
    size=(0.4, 0.2, 0.6),
    name="car",
    children=[]
)
player.add_component("rigidbody", bereshit.Rigidbody(mass=1, useGravity=True, isKinematic=False))

player_body = bereshit.Object(
    position=Vector3(0, 0.3, -floor_length / 2 + 3.0),
    size=(0.4, 0.2, 0.6),
    name="body",
    children=[camera]
)

player_body.add_component("collider", bereshit.BoxCollider())
player_body.add_component("rigidbody", bereshit.Rigidbody(mass=1, useGravity=True, isKinematic=False))
player_body.add_component("playerController", playerController.PlayerController())
# player_body.add_component("joint",bereshit.FixJoint(player))

# --- Player Car ---

# player.add_component("collider", bereshit.BoxCollider())
player.material.kind = "Car"
player.material.color = "blue"
# player.add_component("rigidbody", bereshit.Rigidbody(mass=1, useGravity=True, isKinematic=False))

# --- Finish Line ---
goal = bereshit.Object(
    position=Vector3(0, 0.2, floor_length / 2 - 1.0),
    size=(floor_width, 0.1, 0.2),
    name="finish_line"
)
goal.material.kind = "Finish"
goal.material.color = "yellow"
goal.add_component("collider", bereshit.BoxCollider(is_trigger=True))
goal.add_component("Goal", Goal.Goal())
goal.add_component("rigidbody", bereshit.Rigidbody(mass=0.001, useGravity=False, isKinematic=True))

# camera.add_component("camController", CamController.CamController(player))

# --- Scene ---
scene = bereshit.Object(
    position=Vector3(0, 0, 0),
    size=(0, 0, 0),
    children=[floor, *walls, *obstacles, player_body, player, goal],
    name="scene"
)

world = bereshit.Object(
    position=Vector3(0, 0, 0),
    size=(0, 0, 0),
    children=[scene, camera],
    name="world"
)

Core.run(world)
