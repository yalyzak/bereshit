import math
import time
import numpy as np
from dataclasses import dataclass
import statistics
import copy
import send
from math import sqrt

@dataclass
class Vector3:
    x: float = 0
    y: float = 0
    z: float = 0

    def __iadd__(self, other):
        if isinstance(other, Vector3):
            self.x += other.x
            self.y += other.y
            self.z += other.z
            return self
        raise TypeError(f"UnsuPPO2rted type for +=: {type(other)}")

    def __copy__(self):
        return Vector3(self.x,self.y,self.z)
    def __neg__(self):
        return Rotation(-self.x, -self.y, -self.z)
    def __add__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, (list, tuple, np.ndarray)) and len(other) == 3:
            return Vector3(self.x + other[0], self.y + other[1], self.z + other[2])
        raise TypeError(f"UnsuPPO2rted type for addition: {type(other)}")

    def __sub__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
        raise TypeError("Subtraction only suPPO2rted between Vector3 instances")

    def __truediv__(self, other):
        if isinstance(other, Vector3):
            return Vector3(
                self.x / other.x if other.x != 0 else 0,
                self.y / other.y if other.y != 0 else 0,
                self.z / other.z if other.z != 0 else 0
            )
        elif isinstance(other, (int, float)):
            return Vector3(self.x / other, self.y / other, self.z / other)
        raise TypeError(f"UnsuPPO2rted division between Vector3 and {type(other)}")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector3(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, Vector3):
            return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
        raise TypeError("UnsuPPO2rted type for multiplication")

    def dis(self, other: 'Vector3') -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return sqrt(dx * dx + dy * dy + dz * dz)
    @staticmethod
    def mean(vectors: list):
        n = len(vectors)
        if n == 0:
            return Vector3(0, 0, 0)
        sx = sum(v.x for v in vectors)
        sy = sum(v.y for v in vectors)
        sz = sum(v.z for v in vectors)
        return Vector3(sx / n, sy / n, sz / n)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def magnitude(self):
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

    def normalized(self):
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0)  # Handle zero-length vector safely
        return Vector3(self.x / mag, self.y / mag, self.z / mag)

    def direction_vector(self, other):
        return (other - self).normalized()

    def reduce_vector_along_direction(size_vector, direction_vector):
        dot = size_vector.dot(direction_vector)
        if dot > 0:
            return direction_vector * dot
        else:
            return Vector3(0, 0, 0)

    def __rmul__(self, other):
        return self.__mul__(other)

    def to_tuple(self):
        return (self.x, self.y, self.z)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z


class Position(Vector3): pass


class LocalPosition(Vector3): pass


class CenterOfGravity(Vector3): pass


class Rotation(Vector3): pass


class LocalRotation(Vector3): pass


class Size(Vector3):
    def __init__(self, x=1, y=1, z=1):
        super().__init__(x, y, z)


class BoxCollider:
    def __init__(self, size=Vector3(1, 1, 1), object_pointer=None,is_trigger = False):
        self.size = size
        self.obj = object_pointer
        self.is_trigger = is_trigger

    def OnTriggerEnter(self, other_collider):
        """This method can be overwritten by subclasses to handle trigger events."""
        print(self.obj.name,other_collider.obj.name)
        pass
    def Positional_correction(self, other):
        min_max_a_list = self.get_bounds()
        min_max_b_list = other.get_bounds()

        smallest_translation = None
        smallest_distance = float('inf')

        for min_a, max_a in min_max_a_list:
            for min_b, max_b in min_max_b_list:
                dx = min(max_a.x, max_b.x) - max(min_a.x, min_b.x)
                dy = min(max_a.y, max_b.y) - max(min_a.y, min_b.y)
                dz = min(max_a.z, max_b.z) - max(min_a.z, min_b.z)

                if dx > 0 and dy > 0 and dz > 0:
                    # For each axis, check both direction signs
                    candidates = [
                        Vector3(dx, 0, 0),
                        Vector3(-dx, 0, 0),
                        Vector3(0, dy, 0),
                        Vector3(0, -dy, 0),
                        Vector3(0, 0, dz),
                        Vector3(0, 0, -dz),
                    ]

                    for v in candidates:
                        moved_a_min = min_a + v
                        moved_a_max = max_a + v

                        if not (
                                moved_a_max.x < min_b.x or moved_a_min.x > max_b.x or
                                moved_a_max.y < min_b.y or moved_a_min.y > max_b.y or
                                moved_a_max.z < min_b.z or moved_a_min.z > max_b.z
                        ):
                            continue  # Still overlapping in this direction

                        distance = v.magnitude()
                        if distance < smallest_distance:
                            smallest_translation = v
                            smallest_distance = distance

        if smallest_translation:
            self.obj.position += smallest_translation

    def get_bounds(self):
        # Get 8 corners in local space
        half = self.size * 0.5
        local_corners = [
            Vector3(x, y, z)
            for x in (-half.x, half.x)
            for y in (-half.y, half.y)
            for z in (-half.z, half.z)
        ]

        # Rotate and translate to world space
        world_corners = [rotate_vector(corner, self.obj.position, self.obj.rotation) + self.obj.position for corner in
                         local_corners]

        # Find min/max of all corners
        min_bound = Vector3(
            min(c.x for c in world_corners),
            min(c.y for c in world_corners),
            min(c.z for c in world_corners)
        )
        max_bound = Vector3(
            max(c.x for c in world_corners),
            max(c.y for c in world_corners),
            max(c.z for c in world_corners)
        )
        bounds = []
        for child in self.obj.get_children_bereshit():
            bounds = child.collider.get_bounds()
        bounds.append([min_bound, max_bound])
        return bounds

    def get_box_corners(self):
        center = self.obj.position
        size = self.size
        half = size * 0.5

        # All 8 corner offsets from center (±x, ±y, ±z)
        offsets = [
            Vector3(x, y, z)
            for x in (-half.x, half.x)
            for y in (-half.y, half.y)
            for z in (-half.z, half.z)
        ]

        corners = [rotate_vector(center + offset, center, self.obj.rotation) for offset in offsets]
        return corners

    def check_collision(self, other):
        other_collider = getattr(other, 'collider', other)
        if other_collider is None:
            return None

        # --- Internal Functions ---

        def get_axes(rotation):
            right = rotate_vector(Vector3(1, 0, 0), Vector3(0, 0, 0), rotation).normalized()
            up = rotate_vector(Vector3(0, 1, 0), Vector3(0, 0, 0), rotation).normalized()
            forward = rotate_vector(Vector3(0, 0, 1), Vector3(0, 0, 0), rotation).normalized()
            return [right, up, forward]

        def project_box(center, axes, half_sizes, axis):
            projection_center = center.dot(axis)
            radius = sum(abs(axis.dot(a)) * h for a, h in zip(axes, half_sizes))
            return projection_center - radius, projection_center + radius

        def overlap_on_axis(proj1, proj2):
            return not (proj1[1] < proj2[0] or proj2[1] < proj1[0])

        # --- SAT Collision Detection ---

        a_center = self.obj.position
        b_center = other_collider.obj.position

        a_axes = get_axes(self.obj.rotation)
        b_axes = get_axes(other_collider.obj.rotation)

        a_half = self.obj.size * 0.5
        b_half = other_collider.obj.size * 0.5

        a_half_sizes = [a_half.x, a_half.y, a_half.z]
        b_half_sizes = [b_half.x, b_half.y, b_half.z]

        axes_to_test = []

        # Add 3 axes of A
        axes_to_test.extend(a_axes)

        # Add 3 axes of B
        axes_to_test.extend(b_axes)

        # Add 9 cross-product axes
        for i in range(3):
            for j in range(3):
                cross = a_axes[i].cross(b_axes[j])
                if cross.magnitude() > 1e-6:  # avoid zero/near-zero axes
                    axes_to_test.append(cross.normalized())

        smallest_overlap = float('inf')
        collision_axis = None

        for axis in axes_to_test:
            proj_a = project_box(a_center, a_axes, a_half_sizes, axis)
            proj_b = project_box(b_center, b_axes, b_half_sizes, axis)

            if not overlap_on_axis(proj_a, proj_b):
                return None  # Separating axis found → no collision

            overlap = min(proj_a[1], proj_b[1]) - max(proj_a[0], proj_b[0])
            if overlap < smallest_overlap:
                smallest_overlap = overlap
                collision_axis = axis

        # Estimate contact point and normal
        contact_point = (a_center + b_center) * 0.5
        normal = collision_axis.normalized()

        # Ensure the normal points from B to A
        # Ensure the normal points from B to A
        if (a_center - b_center).dot(normal) < 0:
            normal = normal * -1

        if self.is_trigger:
            self.OnTriggerEnter(other_collider)
        if other_collider.is_trigger:
            other_collider.OnTriggerEnter(self)

        return contact_point, normal, smallest_overlap

    def check_collisionold(self, other):
        # Get collider of the other object
        other_collider = getattr(other, 'collider', other)
        if other_collider is None:
            return None

        # Get the corners of this collider (AABB or OBB transformed)
        corners = self.get_box_corners()




        collided = []
        local_collisions = []


        for corner in corners:
            # Convert to local space of the other object
            local_point = corner - other_collider.obj.position
            local_point = inverse_rotate_vector(local_point, other_collider.obj.rotation)
            half_size = other_collider.obj.size * 0.5

            if (-half_size.x <= local_point.x <= half_size.x and
                    -half_size.y <= local_point.y <= half_size.y and
                    -half_size.z <= local_point.z <= half_size.z):
                collided.append(corner)
                local_collisions.append(local_point)

        if not collided:
            return None

        # Step 1: Compute average world-space contact point
        contact_point = Vector3.mean(collided)

        # Step 2: Estimate normal from average local collision point
        avg_local = Vector3.mean(local_collisions)

        # Determine which face was likely hit based on closest axis
        # (X, Y, Z face based on proximity to +half/-half)
        dx = abs(abs(avg_local.x) - half_size.x)
        dy = abs(abs(avg_local.y) - half_size.y)
        dz = abs(abs(avg_local.z) - half_size.z)

        if dx <= dy and dx <= dz:
            normal_local = Vector3(1 if avg_local.x > 0 else -1, 0, 0)
        elif dy <= dx and dy <= dz:
            normal_local = Vector3(0, 1 if avg_local.y > 0 else -1, 0)
        else:
            normal_local = Vector3(0, 0, 1 if avg_local.z > 0 else -1)

        # Step 3: Rotate normal to world space
        normal_world = rotate_vector(normal_local, Vector3(0, 0, 0), other_collider.obj.rotation)

        return contact_point, normal_world.normalized()


class Rigidbody:
    def __init__(self, obj=None, mass=1.0, size=Vector3(1, 1, 1), position=Vector3(0, 0, 0),
                 center_of_mass=Vector3(0, 0, 0), velocity=None, angular_velocity=Vector3(), force=None,
                 isKinematic=False, useGravity=True, drag=1, friction_coefficient=0.6,restitution=0.3):
        self.mass = mass
        self.obj = obj
        self.drag = drag
        self.restitution = restitution
        self.friction_coefficient = friction_coefficient
        self.center_of_mass = center_of_mass if center_of_mass else position
        self.velocity = velocity or Vector3(0, 0, 0)
        self.acceleration = Vector3()
        self.angular_acceleration = Vector3()
        self.torque = Vector3()
        self.force = force or Vector3(0, 0, 0)
        self.isKinematic = isKinematic
        self.useGravity = useGravity
        self.angular_velocity = angular_velocity
        if self.obj != None:
            self.inertia = Vector3(
                (1 / 12) * self.mass * (self.obj.size.y ** 2 + self.obj.size.z ** 2),  # I_x
                (1 / 12) * self.mass * (self.obj.size.x ** 2 + self.obj.size.z ** 2),  # I_y
                (1 / 12) * self.mass * (self.obj.size.x ** 2 + self.obj.size.y ** 2)  # I_z
            )

    def exert_force(self, force, dt=0.01):
        # F = ma ⇒ a = F/m
        self.force += force
        self.obj.update(dt)
        # self.obj.integrat(dt)

    def accelerate(self, acceleration, dt=0.01):
        self.acceleration += acceleration
        self.obj.integrat2(dt)

    def apply_torque(self, force: Vector3, application_point: Vector3):
        # Offset from center of mass to point of application
        r = application_point - self.obj.position  # self.position = center of mass

        # Torque = r × F (cross product)
        torque = r.cross(force)

        # Add to total torque
        self.torque += torque

    def exert_angular_acceleration(self, force, application_point=Vector3):
        # Offset from center of mass to point of application
        r = application_point - self.obj.position  # r = lever arm from CoM

        # Torque = r × F
        torque = r.cross(force)

        # Angular acceleration = torque / inertia
        self.angular_acceleration += Vector3(
            torque.x / self.inertia.x,
            torque.y / self.inertia.y,
            torque.z / self.inertia.z
        )

import PPO2
class Joint:
    def __init__(self, obj=None, other=None, position=None, rotation=None, look_position=True, look_rotation=False):
        self.obj = obj
        self.other = other
        self.look_position = look_position
        self.look_rotation = look_rotation

class AgentController:
    def __init__(self, obj, goal, obs_dim=6, action_dim=4):
        self.obj = obj              # the object this agent controls
        self.goal = goal            # its target
        self.agent = PPO2.PPO2Agent(obs_dim=obs_dim, action_dim=action_dim)
        self.reset()

    def get_obs(self):
        return [
            self.goal.position.x, self.goal.position.y, self.goal.position.z,
            self.obj.position.x, self.obj.position.y, self.obj.position.z
        ]

    def act(self):
        obs = self.get_obs()
        action, logp, value = self.agent.get_action(obs)
        self.last_obs = obs
        self.last_action = action
        self.last_logp = logp
        self.last_value = value
        self.apply_action(action)

    def apply_action(self, action):
        if action == 0:
            self.obj.rigidbody.velocity.z += 0.5
        elif action == 1:
            self.obj.rigidbody.velocity.x += -0.5
        elif action == 2:
            self.obj.rigidbody.velocity.z += -0.5
        elif action == 3:
            self.obj.rigidbody.velocity.x += 0.5

    def store_experience(self, reward, done):
        self.agent.store((
            self.last_obs,
            self.last_action,
            self.last_logp,
            reward,
            self.last_value,
            done
        ))

    def update(self):
        self.agent.update()

    def reset(self):
        self.last_obs = None
        self.last_action = None
        self.last_logp = None
        self.last_value = None

class Object:
    @property
    def local_position(self):
        if self.parent is None:
            return copy.copy(self.position)

        # Step 1: Offset vector from parent to this object
        offset = self.position - self.parent.position

        # Step 2: Rotate offset into local space (undo parent rotation)
        inverse_rotation = -self.parent.rotation
        local_offset = rotate_vector(offset, Vector3(0, 0, 0), inverse_rotation)

        return local_offset

    def __copy__(self):
        return Object(self.value)

    def __deepcopy__(self, memo):
        obj_copy = type(self)(
            position=copy.deepcopy(self.position, memo),
            rotation=copy.deepcopy(self.rotation, memo),
            size=copy.deepcopy(self.size, memo),
            children=copy.deepcopy(self.children, memo),
            components=copy.deepcopy(self.components, memo),
            name=copy.deepcopy(self.name, memo),
        )
        memo[id(self)] = obj_copy

        # Fix parent for children
        for child in obj_copy.children:
            if isinstance(child, Object):
                child.parent = obj_copy
            elif hasattr(child, 'obj') and isinstance(child.obj, Object):
                child.obj.parent = obj_copy

        # Fix component references
        for comp in obj_copy.components.values():
            if hasattr(comp, 'obj'):
                comp.obj = obj_copy

        return obj_copy

    def __init__(self, position=None, rotation=None, size=None, children=None, components=None,
                 name=""):
        self.parent = None
        self.children = children or []
        self.name = name
        self.size = Size(*size) if isinstance(size, tuple) else size or Size()
        self.components = components or {}

        self.local_rotation = LocalRotation()

        for child in self.children:
            if isinstance(child, Object):
                child.parent = self
            elif isinstance(child, Servo):
                child.obj.parent = self
        self.position = Position(*position) if isinstance(position, tuple) else position or Position()

        self.__default_position = self.position
        self.rotation = Rotation(*rotation) if isinstance(rotation, tuple) else rotation or Rotation()
        # self.collider = BoxCollider(self.size, self) if size else None
        self.world = None



        def findWorld(child):
            parent = child.parent
            if parent is None:
                return child

            return findWorld(parent)

        for child in self.get_all_children_bereshit():
            child.world = findWorld(child)




    def search(self, target_name):
        if hasattr(self, 'name') and self.name == target_name:
            return self
        if hasattr(self, 'children'):
            for child in self.children:
                result = child.search(target_name)  # ✅ fix here
                if result:
                    return result
        return None

    def add_component(self, name, component):
        if name == "rigidbody":
            if isinstance(component, tuple):
                rb = Rigidbody(*component)
            elif isinstance(component, Rigidbody):
                rb = component
            else:
                raise TypeError("Invalid type for Rigidbody")

            # Configure based on Object
            rb.size = self.size
            rb.position = self.position
            rb.center_of_mass = self.position
            rb.obj = self
            rb.inertia = Vector3(
                (1 / 12) * rb.mass * (rb.obj.size.y ** 2 + rb.obj.size.z ** 2),  # I_x
                (1 / 12) * rb.mass * (rb.obj.size.x ** 2 + rb.obj.size.z ** 2),  # I_y
                (1 / 12) * rb.mass * (rb.obj.size.x ** 2 + rb.obj.size.y ** 2)  # I_z
            )
            component = rb
        elif name == "collider":
            if isinstance(component, tuple):
                boxcollider = BoxCollider(*component)
            elif isinstance(component, BoxCollider):
                boxcollider = component
            else:
                raise TypeError("Invalid type for BoxCollider")

            # Configure based on Object
            boxcollider.size = self.size
            boxcollider.obj = self
            # boxcollider.is_trigger = False
            component = boxcollider
        elif name == "joint":
            if isinstance(component, tuple):
                joint = Joint(*component)
            elif isinstance(component, Joint):
                joint = component
            else:
                raise TypeError("Invalid type for Joint")
        elif name == "agent":
            if isinstance(component, tuple):
                agent = PPO2.Agent(*component)
            elif isinstance(component, PPO2.Agent):
                agent = component
            else:
                raise TypeError("Invalid type for Agent")
        # Save component
        self.components[name] = component
        component.parent = self  # optional back-reference

    def remove_component(self, name):
        if name in self.components:
            del self.components[name]

    def get_component(self, name):
        return self.components.get(name, None)

    def __getattr__(self, name):
        # Allow normal attribute access
        if hasattr(type(self), name):
            return object.__getattribute__(self, name)
        # Only called if normal attribute lookup fails
        component = self.components.get(name)
        if component is not None:
            return component
        raise AttributeError(f"'{self.name}' object has no attribute or component '{name}'")

    # @property
    # def position(self) -> Vector3:
    #     if self.parent is None:
    #         return self.local_position
    #
    #     # --- Build parent's rotation matrix (Z·Y·X order) ---
    #     ang = np.radians([self.parent.rotation.x,
    #                       self.parent.rotation.y,
    #                       self.parent.rotation.z])
    #     cx, cy, cz = np.cos(ang)
    #     sx, sy, sz = np.sin(ang)
    #
    #     R_x = np.array([[1, 0, 0],
    #                     [0, cx, -sx],
    #                     [0, sx, cx]])
    #     R_y = np.array([[cy, 0, sy],
    #                     [0, 1, 0],
    #                     [-sy, 0, cy]])
    #     R_z = np.array([[cz, -sz, 0],
    #                     [sz, cz, 0],
    #                     [0, 0, 1]])
    #     R_parent = R_z @ R_y @ R_x
    #
    #     lp = np.array([self.local_position.x,
    #                    self.local_position.y,
    #                    self.local_position.z])
    #     rotated = R_parent @ lp
    #
    #     px = self.parent.position.x
    #     py = self.parent.position.y
    #     pz = self.parent.position.z
    #     return Vector3(rotated[0] + px,
    #                     rotated[1] + py,
    #                     rotated[2] + pz)
    #
    # @position.setter
    # def position(self, value: Vector3):
    #     if self.parent is None:
    #         self.local_position = value
    #     else:
    #         # Invert parent's rotation to go from world → parent local
    #         ang = np.radians([self.parent.rotation.x,
    #                           self.parent.rotation.y,
    #                           self.parent.rotation.z])
    #         cx, cy, cz = np.cos(ang)
    #         sx, sy, sz = np.sin(ang)
    #
    #         R_x = np.array([[1, 0, 0],
    #                         [0, cx, -sx],
    #                         [0, sx, cx]])
    #         R_y = np.array([[cy, 0, sy],
    #                         [0, 1, 0],
    #                         [-sy, 0, cy]])
    #         R_z = np.array([[cz, -sz, 0],
    #                         [sz, cz, 0],
    #                         [0, 0, 1]])
    #         R_parent = R_z @ R_y @ R_x
    #         R_inv = np.linalg.inv(R_parent)
    #
    #         wp = np.array([value.x, value.y, value.z])
    #         pp = np.array([self.parent.position.x,
    #                        self.parent.position.y,
    #                        self.parent.position.z])
    #         local_vec = R_inv @ (wp - pp)
    #         self.local_position = Vector3(local_vec[0],
    #                                        local_vec[1],
    #                                        local_vec[2])

    def rotate_around_axis(self, axis, angle_rad):
        """
        Rotates the object around a given axis by angle_rad (in radians).
        """
        axis = axis.normalized()

        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        ux, uy, uz = axis.x, axis.y, axis.z

        # Rodrigues' rotation formula
        R = np.array([
            [cos_a + ux ** 2 * (1 - cos_a), ux * uy * (1 - cos_a) - uz * sin_a, ux * uz * (1 - cos_a) + uy * sin_a],
            [uy * ux * (1 - cos_a) + uz * sin_a, cos_a + uy ** 2 * (1 - cos_a), uy * uz * (1 - cos_a) - ux * sin_a],
            [uz * ux * (1 - cos_a) - uy * sin_a, uz * uy * (1 - cos_a) + ux * sin_a, cos_a + uz ** 2 * (1 - cos_a)]
        ])

        if not hasattr(self, 'rotation_matrix'):
            self.rotation_matrix = np.eye(3)

        self.rotation_matrix = R @ self.rotation_matrix

    def get_all_colliders(self):
        all_bereshit = []
        if self.collider:
            all_bereshit.append(self)
        for child in self.get_all_children_bereshit():
            all_bereshit.extend(child.get_all_colliders())
        return all_bereshit

    def update(self, dt=0.01):
        rb = self.get_component("rigidbody")
        children = self.get_all_children_bereshit()

        center_of_mass = [self.position]
        if rb is not None:
            for child in children:
                center_of_mass.append(child.position)
            self.rigidbody.center_of_mass = Vector3.mean(center_of_mass)

        if self.parent == None:  # the world does not need an update
            for child in children:
                child.update(dt=dt)
            return

        if rb is None or rb.isKinematic or self.get_component("collider") is None:
            return

        # # === 1) RESET ALL FORCES & TORQUES ===
        # self.rigidbody.force = Vector3(0, 0, 0)
        # self.rigidbody.torque = Vector3(0, 0, 0)

        # === 2) APPLY GRAVITY (AND TORSOUE DUE TO GRAVITY) ===
        if self.rigidbody.useGravity:
            gravity = Vector3(0, -9.8, 0)
            self.rigidbody.force += gravity * self.rigidbody.mass

            # If you want torque from gravity:
            r = self.position - self.rigidbody.center_of_mass
            gravity_torque = r.cross(gravity * self.rigidbody.mass)
            self.rigidbody.torque += gravity_torque

        # === 3) COLLISION LOOP ===
        # collisions_processed = set()
        for child in self.world.get_all_children_bereshit():
            if child != self:
                if child not in children and child != self.parent and child.get_component("collider") is not None:
                    # if id(child) in collisions_processed:
                    #     continue

                    result = self.collider.check_collision(child)
                    if result is None:
                        continue  # No collision
                    contact_point, normal = result#child.name == 'wall'
                    normal_force = self.rigidbody.force.reduce_vector_along_direction(normal * -1) * -1

                    # normal2 = contact_point.direction_vector(self.position)
                    # Mark as processed
                    # collisions_processed.add(id(child))

                    if child.get_component("rigidbody") is None or child.rigidbody.isKinematic:
                        self.resolve_kinematic_collision(child, normal, contact_point, normal_force)
                    else:
                        self.resolve_dynamic_collision(child, normal, contact_point, normal_force)
                    # === After updating velocity with all forces ===

                    self.apply_friction(normal_force, dt)


                    # self.integrat(dt)
        self.integrat(dt)
        for child in children:
            child.update(dt=dt)
        # === 1) RESET ALL FORCES & TORQUES ===
        self.rigidbody.force = Vector3(0, 0, 0)
        self.rigidbody.torque = Vector3(0, 0, 0)

    def resolve_kinematic_collision(self, child, normal, contact_point, normal_force):
        # self.collider.Positional_correction(child.collider)

        # normal_force = rotate_vector(vector=gravity * self.rigidbody.mass * -1, pivot=Vector3(0, 0, 0),
        #                              angles=child.rotation)
        # normal_force = rotate_vector(vector=self.rigidbody.force * -1, pivot=Vector3(0, 0, 0),
        #                              angles=direction_to_angles(normal))
        self.rigidbody.force += normal_force
        if child.get_component("rigidbody") is None:
            e = self.rigidbody.restitution  # Coefficient of restitution (e = 0: perfectly inelastic, e = 1: elastic)
            v2_n = 0  # still allowed, but v2_n should be constant
            v1_n = self.rigidbody.velocity.dot(normal)  # normal component of self's velocity
            relative_velocity = self.rigidbody.velocity - Vector3(0,0,0)

        else:
            e = child.rigidbody.restitution  # Coefficient of restitution (e = 0: perfectly inelastic, e = 1: elastic)
            v2_n = child.rigidbody.velocity.dot(normal)  # still allowed, but v2_n should be constant
            v1_n = self.rigidbody.velocity.dot(normal)  # normal component of self's velocity
            relative_velocity = self.rigidbody.velocity - child.rigidbody.velocity


        if relative_velocity.dot(normal) < 0:
            # === Linear velocity update ===
            # Compute new normal velocity after bounce
            new_v1_n = -e * (
                    v1_n - v2_n) + v2_n  # This preserves relative speed and direction based on e

            # Apply delta only to self (child is kinematic)
            delta_v_n = (new_v1_n - v1_n) * normal
            self.rigidbody.velocity += delta_v_n

            # === Angular velocity update ===
            r = contact_point - self.rigidbody.center_of_mass
            impulse = self.rigidbody.mass * delta_v_n
            angular_impulse = r.cross(impulse)
            # self.rigidbody.angular_velocity += angular_impulse / self.rigidbody.inertia

        r = contact_point - self.rigidbody.center_of_mass
        torque = r.cross(normal_force)
        # self.rigidbody.torque += torque

    def resolve_dynamic_collision(self, child, normal, contact_point, normal_force):
        # Compute normal force to cancel the penetration force between two dynamic bereshit



        self.rigidbody.force += normal_force

        child.rigidbody.force -= normal_force  # Equal and oPPO2site

        e = min(self.rigidbody.restitution, child.rigidbody.restitution)

        v1_n = self.rigidbody.velocity.dot(normal)

        v2_n = child.rigidbody.velocity.dot(normal)

        relative_velocity = self.rigidbody.velocity - child.rigidbody.velocity

        if relative_velocity.dot(normal) < 0:
            # Total mass

            m1 = self.rigidbody.mass

            m2 = child.rigidbody.mass

            # Compute impulse scalar

            j = -(1 + e) * (v1_n - v2_n) / (1 / m1 + 1 / m2)

            impulse = j * normal

            # Linear velocity update

            self.rigidbody.velocity += impulse / m1

            child.rigidbody.velocity -= impulse / m2

            # Angular velocity update

            r1 = contact_point - self.rigidbody.center_of_mass

            r2 = contact_point - child.rigidbody.center_of_mass

            angular_impulse1 = r1.cross(impulse)

            angular_impulse2 = r2.cross(impulse * -1)

            # self.rigidbody.angular_velocity += angular_impulse1 / self.rigidbody.inertia

            # child.rigidbody.angular_velocity += angular_impulse2 / child.rigidbody.inertia

        # Torque from normal force

        r1 = contact_point - self.rigidbody.center_of_mass

        r2 = contact_point - child.rigidbody.center_of_mass

        torque1 = r1.cross(normal_force)

        torque2 = r2.cross(normal_force * -1)

        # self.rigidbody.torque += torque1

        # child.rigidbody.torque += torque2

    def apply_friction(self, normal_force, dt):
        rb = self.rigidbody
        if rb.velocity.magnitude() > 0:
            d = rb.velocity.normalized() * -1
            friction_acc = d * (normal_force.magnitude() * rb.friction_coefficient / rb.mass)
            new_velocity = rb.velocity + friction_acc * dt
            if new_velocity.dot(rb.velocity) <= 0:
                rb.velocity = Vector3(0, 0, 0)
            else:
                rb.velocity = new_velocity

        if rb.force.magnitude() > 0:
            d = rb.force.normalized() * -1
            friction_magnitude = normal_force.magnitude() * rb.friction_coefficient
            applied_force_magnitude = rb.force.magnitude()
            max_friction = min(friction_magnitude, applied_force_magnitude)
            rb.force += d * max_friction

    def integrat(self, dt):

        # === 4) INTEGRATION PHASE ===
        # 4.1) Linear acceleration & velocity:
        self.rigidbody.acceleration = self.rigidbody.force / self.rigidbody.mass
        self.rigidbody.acceleration.y *= self.rigidbody.drag

        self.rigidbody.velocity += self.rigidbody.acceleration * dt
        # 4.2) Angular acceleration & velocity (component‐wise):
        self.rigidbody.angular_acceleration = Vector3(
            self.rigidbody.torque.x / self.rigidbody.inertia.x if self.rigidbody.inertia.x != 0 else 0,
            self.rigidbody.torque.y / self.rigidbody.inertia.y if self.rigidbody.inertia.y != 0 else 0,
            self.rigidbody.torque.z / self.rigidbody.inertia.z if self.rigidbody.inertia.z != 0 else 0
        )
        self.rigidbody.angular_velocity += self.rigidbody.angular_acceleration * dt

        # 4.3) Integrate rotation
        ang_disp = self.rigidbody.angular_velocity * dt \
                   + 0.5 * self.rigidbody.angular_acceleration * dt * dt

        if self.get_component("joint") != None:
            if not self.joint.look_position:
                # 4.4) Integrate position

                self.position += self.rigidbody.velocity * dt \
                                 + 0.5 * self.rigidbody.acceleration * dt * dt
                # self.set_position(self.position)
            if not self.joint.look_rotation:
                self.add_rotation(ang_disp)
        else:
            # 4.4) Integrate position
            self.position += self.rigidbody.velocity * dt \
                             + 0.5 * self.rigidbody.acceleration * dt * dt
            for
            # self.set_position(self.position)

            self.add_rotation(ang_disp)

        # 4.5) Angular damping
        # self.rigidbody.angular_velocity *= 0.98

        # 4.6) Reset torques for next frame
        self.rigidbody.angular_acceleration = Vector3(0, 0, 0)
        self.rigidbody.torque = Vector3(0, 0, 0)
    def integrat2(self, dt):

        # === 4) INTEGRATION PHASE ===
        self.rigidbody.velocity += self.rigidbody.acceleration * dt
        # 4.2) Angular acceleration & velocity (component‐wise):
        self.rigidbody.angular_acceleration = Vector3(
            self.rigidbody.torque.x / self.rigidbody.inertia.x if self.rigidbody.inertia.x != 0 else 0,
            self.rigidbody.torque.y / self.rigidbody.inertia.y if self.rigidbody.inertia.y != 0 else 0,
            self.rigidbody.torque.z / self.rigidbody.inertia.z if self.rigidbody.inertia.z != 0 else 0
        )
        self.rigidbody.angular_velocity += self.rigidbody.angular_acceleration * dt

        # 4.3) Integrate rotation
        ang_disp = self.rigidbody.angular_velocity * dt \
                   + 0.5 * self.rigidbody.angular_acceleration * dt * dt
        if self.get_component("joint") != None:
            if not self.joint.look_position:
                # 4.4) Integrate position

                self.position += self.rigidbody.velocity * dt \
                                 + 0.5 * self.rigidbody.acceleration * dt * dt
                self.set_position(self.position)
            if not self.joint.look_rotation:
                self.add_rotation(ang_disp)
        else:
            # 4.4) Integrate position
            self.position += self.rigidbody.velocity * dt \
                             + 0.5 * self.rigidbody.acceleration * dt * dt

            self.set_position(self.position)

            self.add_rotation(ang_disp)

        # 4.5) Angular damping
        # self.rigidbody.angular_velocity *= 0.98

        # 4.6) Reset torques for next frame
        self.rigidbody.angular_acceleration = Vector3(0, 0, 0)
        self.rigidbody.torque = Vector3(0, 0, 0)

    def set_local_position(self):
        for child in self.children:
            target = child.obj if isinstance(child, Servo) else child
            target.local_position = target.position - target.parent.position
            target.set_local_position()

    def set_local_rotation(self):
        for child in self.children:
            target = child.obj if isinstance(child, Servo) else child
            target.local_rotation = target.rotation - target.parent.rotation
            target.set_local_rotation()

    def set_position(self, new_position):
        delta = new_position - self.position
        self.position += delta
        self._set_local_position_recursive(delta)

    def _set_local_position_recursive(self, delta):
        for child in self.children:
            target = child.obj if isinstance(child, Servo) else child
            target.position += delta
            target._set_local_position_recursive(delta)  # Recurse into grandchildren

            # target.local_rotation = target.rotation - target.parent.rotation
            # target.set_local_rotation()

    def set_rotation(self, new_world_rot: Vector3):
        """
        Explicitly set this object's world‐space rotation to `new_world_rot`.
        Also recompute `local_rotation` so that:
            local_rotation = new_world_rot - parent.rotation
        (or = new_world_rot if there's no parent).
        """
        # 1. Assign the new world rotation
        self.rotation = Vector3(new_world_rot.x, new_world_rot.y, new_world_rot.z)

        # 2. Compute and store the local offset from parent
        if self.parent is not None:
            self.local_rotation = Vector3(
                self.rotation.x - self.parent.rotation.x,
                self.rotation.y - self.parent.rotation.y,
                self.rotation.z - self.parent.rotation.z
            )
        else:
            # No parent means local == world
            self.local_rotation = Vector3(self.rotation.x,
                                           self.rotation.y,
                                           self.rotation.z)

    def add_rotation(self, delta: Vector3):
        """
        Add an incremental rotation `delta` (world‐space Euler angles)
        on top of the current world rotation. Then update local_rotation.
        """
        # 1. Compute the new world rotation by simple vector addition
        new_world_rot = Vector3(
            self.rotation.x + delta.x,
            self.rotation.y + delta.y,
            self.rotation.z + delta.z
        )

        # 2. Delegate to set_rotation to keep local_rotation in sync
        self.set_rotation(new_world_rot)

    def add_rotation_old(self, rotation):
        self.set_rotation(self.rotation + rotation)

    def set_rotation_old(self, rotation):
        delta = rotation - self.rotation
        if delta != Vector3(0, 0, 0):
            self.rotation = rotation
            self.local_rotation = rotation - self.parent.rotation
            self._set_rotation_recursive(delta)
            # self.set_projection()

    def _set_rotation_recursive(self, delta):
        for child in self.children:
            target = child.obj if isinstance(child, Servo) else child
            target.rotation += delta
            target._set_rotation_recursive(delta)

    def set_projection(self, pivot=np.array([0, 0, 0])):
        for child in self.children:
            target = child.obj if isinstance(child, Servo) else child
            vector = np.array([target.position.x, target.position.y, target.position.z])
            if np.allclose(pivot, [0, 0, 0]):
                pivot = np.array([target.parent.position.x, target.parent.position.y, target.parent.position.z])
            angles = (target.parent.local_rotation.x, target.parent.local_rotation.y, target.parent.local_rotation.z)
            rotated = rotate_vector(vector, pivot, angles)
            # target.set_position(Vector3(*rotated))
            target.position = Vector3(*rotated)
            target.set_projection(pivot=pivot)

    def find_center_of_gravity(self):
        bereshit = self.get_all_children_bereshit()
        positions = [obj.position.to_tuple() for obj in bereshit]
        masses = [obj.rigidbody.mass for obj in bereshit]
        x_cog, y_cog, z_cog = calculate_center_of_gravity_3d(positions, masses)
        print(f"Center of Gravity: ({x_cog:.2f}, {y_cog:.2f}, {z_cog:.2f})")

    def add_servo(self, servo):
        if isinstance(servo, Servo):
            self.children.append(servo)
        else:
            raise ValueError("Servo must be an instance of the Servo class")

    def reset_to_default(self):
        self.position = self.__default_position
        if self.get_component("rigidbody") is not None:
            self.rigidbody.acceleration = Vector3(0,0,0)
            self.rigidbody.velocity = Vector3(0,0,0)

        for child in self.children:
            child.reset_to_default()

    def rotate_point(self):
        for child in self.children:
            child.rotate_point()

    def get_children_bereshit(self):
        return [child.obj if isinstance(child, Servo) else child for child in self.children]

    def get_all_children_bereshit(self):
        all_objs = []
        for child in self.children:
            target = child.obj if isinstance(child, Servo) else child
            all_objs.append(target)
            all_objs.extend(target.get_all_children_bereshit())
        return all_objs

    def __repr__(self):
        children_repr = ",\n    ".join(repr(child) for child in self.children)
        return (f"Object(\n"
                f"  Position={self.position},\n"
                f"  Rotation={self.rotation},\n"
                f"  Size={self.size},\n"
                f"  children=[\n    {children_repr}\n  ]\n"
                f")")


class Servo:
    def __init__(self, board_number=None, servo_number=None, name=None, default_angle=None, default_pulse=None,
                 direction=" ", obj=None, min_angle=0, max_angle=180):
        self.board_number = board_number
        self.servo_number = servo_number
        self.name = name
        self.direction = direction
        self.obj = obj or Object()
        self.obj.name = name
        self.angle = 0
        self.min_angle = min_angle
        self.max_angle = max_angle

        if default_angle is not None:
            self.default_angle = default_angle
            self.default_pulse = self.angle_to_pulse(default_angle)
        elif default_pulse is not None:
            self.default_pulse = default_pulse
            self.default_angle = self.pulse_to_angle(default_pulse)
        else:
            self.default_angle = 90.0
            self.default_pulse = self.angle_to_pulse(90.0)

        self.current_angle = self.default_angle
        self.current_pulse = self.default_pulse

    def rotate_point(self):
        x, y = self.obj.local_position.x, self.obj.local_position.y + self.obj.size.y / 2
        angle_radians = math.radians(self.angle)
        cos_theta = math.cos(angle_radians)
        sin_theta = math.sin(angle_radians)
        x_new = x * cos_theta - y * sin_theta
        y_new = x * sin_theta + y * cos_theta
        return (x_new, y_new)

    @staticmethod
    def angle_to_pulse(angle):
        return int(400 + (angle / 180) * (2750 - 400))

    @staticmethod
    def pulse_to_angle(pulse):
        return (pulse - 400) * 180 / (2750 - 400)

    @staticmethod
    def direction_to_vector(direction, delta):
        return {
            "+z": Vector3(delta, 0, 0),
            "-z": Vector3(-delta, 0, 0),
            "+y": Vector3(0, delta, 0),
            "-y": Vector3(0, -delta, 0),
            "+x": Vector3(0, 0, delta),
            "-x": Vector3(0, 0, -delta),
        }.get(direction, Vector3(0, 0, 0))

    def set_angle(self, angle):
        delta = angle - self.current_angle
        angle = max(self.min_angle, min(angle, self.max_angle))

        self.current_angle = angle
        self.current_pulse = self.angle_to_pulse(angle)
        self.angle = angle - self.default_angle
        self.obj.add_rotation(self.direction_to_vector(self.direction, delta))
        send.send_message(self.board_number, self.servo_number, self.current_pulse)

    def add_angle(self, angle):
        self.set_angle(self.current_angle + angle)

    def active(self, on):
        send.send_message(self.board_number, self.servo_number, self.current_pulse, header3=on)

    def set_pulse(self, pulse):
        self.current_pulse = pulse
        self.current_angle = self.pulse_to_angle(pulse)
        self.angle = self.current_angle - self.default_angle
        send.send_message(self.board_number, self.servo_number, self.current_pulse)
        print(f"Setting {self.name} to {pulse} µs ({self.current_angle:.1f} degrees).")

    def reset_to_default(self):
        self.current_angle = self.default_angle
        self.current_pulse = self.default_pulse
        send.send_message(self.board_number, self.servo_number, self.default_pulse)
        self.obj.reset_to_default()

    def __repr__(self):
        return (f"Servo(\n"
                f"  board_number={self.board_number},\n"
                f"  servo_number={self.servo_number},\n"
                f"  name={self.name},\n"
                f"  default_angle={self.default_angle},\n"
                f"  default_pulse={self.default_pulse},\n"
                f"  current_angle={self.current_angle},\n"
                f"  current_pulse={self.current_pulse},\n"
                f"  direction={self.direction}\n"
                f")")


def rotate_vector(vector, pivot, angles):
    vector = np.array([vector.x, vector.y, vector.z])
    pivot = np.array([pivot.x, pivot.y, pivot.z])
    angles = np.array([angles.x, angles.y, angles.z])

    angle_x, angle_y, angle_z = np.radians(angles)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(angle_x), -np.sin(angle_x)],
                    [0, np.sin(angle_x), np.cos(angle_x)]])
    R_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y)]])
    R_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                    [np.sin(angle_z), np.cos(angle_z), 0],
                    [0, 0, 1]])
    R = R_z @ R_y @ R_x
    temp = R @ (vector - pivot) + pivot

    return Vector3(temp[0], temp[1], temp[2])


def inverse_rotate_vector(vector, rotation_angles):
    inverse_angles = Vector3(-rotation_angles.x, -rotation_angles.y, -rotation_angles.z)
    return rotate_vector(vector, Vector3(0, 0, 0), inverse_angles)


def move_multiple_servos_degrees(channels, start_list, end_list, steps=None, delay=30):
    if steps is None:
        steps = int(max(abs(end - start) for start, end in zip(start_list, end_list)))
    for step in range(steps + 1):
        for channel, start, end in zip(channels, start_list, end_list):
            current = start + step * (end - start) / steps if steps != 0 else start
            channel.set_angle(current)
        time.sleep(delay / 1000.0)


def calculate_center_of_gravity_3d(positions, masses):
    positions = np.array(positions)
    masses = np.array(masses)
    if len(positions) != len(masses):
        raise ValueError("The number of positions must match the number of masses.")
    total_mass = np.sum(masses)
    x_cog = np.sum(positions[:, 0] * masses) / total_mass
    y_cog = np.sum(positions[:, 1] * masses) / total_mass
    z_cog = np.sum(positions[:, 2] * masses) / total_mass
    return x_cog, y_cog, z_cog

def direction_to_angles(v: Vector3):
    # Prevent division by zero
    if v.magnitude() == 0:
        raise ValueError("Zero direction vector")

    # Normalize first (optional if already normalized)
    v = v / v.magnitude()

    # Yaw (horizontal angle from +Z, around Y axis)
    yaw = math.degrees(math.atan2(v.x, v.z))

    # Pitch (vertical angle from horizontal plane, around X axis)
    pitch = math.degrees(math.asin(v.y)) -90 # Use -v.y if Y is up

    # No roll — direction vectors don’t contain roll info
    return Vector3(pitch, yaw, 0)  # (pitch, yaw, roll)