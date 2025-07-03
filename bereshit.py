import math
import time
import numpy as np
from dataclasses import dataclass
import statistics
import copy
import send
from math import sqrt
import keyboard
dt = 1/60
class camera:
    def __init__(self,width = 1920, hight =1080, FOV = 120,VIEWER_DISTANCE = 0):
        self.width = width
        self.hight = hight
        self.FOV = FOV
        self.VIEWER_DISTANCE = VIEWER_DISTANCE

@dataclass
class Vector3D:

    x: float = 0
    y: float = 0
    z: float = 0

    def floor(self):
        factor = 10 ** 5
        return Vector3D(math.floor(self.x * factor) / factor,math.floor(self.y * factor) / factor,math.floor(self.x * factor) / factor)
    def __iadd__(self, other):
        if isinstance(other, Vector3D):
            self.x += other.x
            self.y += other.y
            self.z += other.z
        elif isinstance(other, (list, tuple, np.ndarray)):
            self.x += other[0]
            self.y += other[1]
            self.z += other[2]
        else:
            raise TypeError(f"Unsupported type for +=: {type(other)}")
        return self


    def __neg__(self):
        return Vector3D(-self.x, -self.y, -self.z)
    def __add__(self, other):
        if isinstance(other, Vector3D):
            return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, (list, tuple, np.ndarray)) and len(other) == 3:
            return Vector3D(self.x + other[0], self.y + other[1], self.z + other[2])
        raise TypeError(f"Unsupported type for addition: {type(other)}")

    def __sub__(self, other):
        if isinstance(other, Vector3D):
            return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
        raise TypeError("Subtraction only supported between Vector3D instances")

    def __truediv__(self, other):
        if isinstance(other, Vector3D):
            if other.x == 0 or other.y == 0 or other.z == 0:
                raise ZeroDivisionError("Division by zero in vector components")

            return Vector3D(
                self.x / other.x if other.x != 0 else 0,
                self.y / other.y if other.y != 0 else 0,
                self.z / other.z if other.z != 0 else 0
            )
        elif isinstance(other, (int, float)):
            return Vector3D(self.x / other, self.y / other, self.z / other)
        raise TypeError(f"Unsupported division between Vector3D and {type(other)}")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector3D(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, Vector3D):
            return Vector3D(self.x * other.x, self.y * other.y, self.z * other.z)
        raise TypeError("Unsupported type for multiplication")
    def __copy__(self):
        return Vector3D(self.x,self.y,self.z)
    def dis(self, other: 'Vector3D') -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return sqrt(dx * dx + dy * dy + dz * dz)
    @staticmethod
    def mean(vectors: list):
        n = len(vectors)
        if n == 0:
            return Vector3D(0, 0, 0)
        sx = sum(v.x for v in vectors)
        sy = sum(v.y for v in vectors)
        sz = sum(v.z for v in vectors)
        return Vector3D(sx / n, sy / n, sz / n)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def magnitude(self):
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

    def normalized(self):
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)  # Handle zero-length vector safely
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)

    def direction_vector(self, other):
        return (other - self).normalized()

    def reduce_vector_along_direction(size_vector, direction_vector):
        dot = size_vector.dot(direction_vector)
        if dot > 0:
            return direction_vector * dot
        else:
            return Vector3D(0, 0, 0)

    def __rmul__(self, other):
        return self.__mul__(other)

    def to_tuple(self):
        return (self.x, self.y, self.z)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z
class Vector4:
    def __init__(self, x=0, y=0, z=0, w=0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __iadd__(self, other):
        if isinstance(other, Vector4):
            self.x += other.x
            self.y += other.y
            self.z += other.z
            self.w += other.w
            return self
        elif isinstance(other, (list, tuple)) and len(other) == 4:
            self.x += other[0]
            self.y += other[1]
            self.z += other[2]
            self.w += other[3]
            return self
        raise TypeError(f"Unsupported type for +=: {type(other)}")

    def __copy__(self):
        return Vector4(self.x, self.y, self.z, self.w)

    def __neg__(self):
        return Vector4(-self.x, -self.y, -self.z, -self.w)

    def __add__(self, other):
        if isinstance(other, Vector4):
            return Vector4(
                self.x + other.x,
                self.y + other.y,
                self.z + other.z,
                self.w + other.w
            )
        elif isinstance(other, (list, tuple)) and len(other) == 4:
            return Vector4(
                self.x + other[0],
                self.y + other[1],
                self.z + other[2],
                self.w + other[3]
            )
        raise TypeError(f"Unsupported type for addition: {type(other)}")

    def __sub__(self, other):
        if isinstance(other, Vector4):
            return Vector4(
                self.x - other.x,
                self.y - other.y,
                self.z - other.z,
                self.w - other.w
            )
        elif isinstance(other, (list, tuple)) and len(other) == 4:
            return Vector4(
                self.x - other[0],
                self.y - other[1],
                self.z - other[2],
                self.w - other[3]
            )
        raise TypeError("Subtraction only supported between Vector4 instances or 4-element sequences")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector4(
                self.x * other,
                self.y * other,
                self.z * other,
                self.w * other
            )
        elif isinstance(other, Vector4):
            return Vector4(
                self.x * other.x,
                self.y * other.y,
                self.z * other.z,
                self.w * other.w
            )
        raise TypeError("Unsupported type for multiplication")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Vector4(
                self.x / other,
                self.y / other,
                self.z / other,
                self.w / other
            )
        elif isinstance(other, Vector4):
            return Vector4(
                self.x / other.x if other.x != 0 else 0,
                self.y / other.y if other.y != 0 else 0,
                self.z / other.z if other.z != 0 else 0,
                self.w / other.w if other.w != 0 else 0
            )
        raise TypeError("Unsupported type for division")

    def __repr__(self):
        return f"Vector4({self.x}, {self.y}, {self.z}, {self.w})"

    def conjugate(self):
        return Vector4(self.w, -self.x, -self.y, -self.z)

    def multiply(self, other):
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return Vector4(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        )

class Position(Vector3D): pass


class LocalPosition(Vector3D): pass


class CenterOfGravity(Vector3D): pass


class Rotation(Vector3D): pass


class LocalRotation(Vector3D): pass


class Size(Vector3D):
    def __init__(self, x=1, y=1, z=1):
        super().__init__(x, y, z)


class BoxCollider:
    def __init__(self, size=Vector3D(1, 1, 1), object_pointer=None,is_trigger = False):
        self.size = size
        self.obj = object_pointer
        self.is_trigger = is_trigger

    def OnTriggerEnter(self, other_collider):
        """This method can be overwritten by subclasses to handle trigger events."""
        for component in self.parent.components.values():
            if hasattr(component, 'OnTriggerEnter') and component.OnTriggerEnter is not None and component != self:
                component.OnTriggerEnter(other_collider)
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
                        Vector3D(dx, 0, 0),
                        Vector3D(-dx, 0, 0),
                        Vector3D(0, dy, 0),
                        Vector3D(0, -dy, 0),
                        Vector3D(0, 0, dz),
                        Vector3D(0, 0, -dz),
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
            Vector3D(x, y, z)
            for x in (-half.x, half.x)
            for y in (-half.y, half.y)
            for z in (-half.z, half.z)
        ]

        # Rotate and translate to world space
        world_corners = [rotate_vector_old(corner, self.obj.position, self.obj.rotation) + self.obj.position for corner in
                         local_corners]

        # Find min/max of all corners
        min_bound = Vector3D(
            min(c.x for c in world_corners),
            min(c.y for c in world_corners),
            min(c.z for c in world_corners)
        )
        max_bound = Vector3D(
            max(c.x for c in world_corners),
            max(c.y for c in world_corners),
            max(c.z for c in world_corners)
        )
        bounds = []
        for child in self.obj.get_children_bereshit():
            bounds = child.collider.get_bounds()
        bounds.append([min_bound, max_bound])
        return bounds

    # def get_box_corners(self):
    #     center = self.obj.position
    #     size = self.size
    #     half = size * 0.5
    #
    #     # All 8 corner offsets from center (±x, ±y, ±z)
    #     offsets = [
    #         Vector3D(x, y, z)
    #         for x in (-half.x, half.x)
    #         for y in (-half.y, half.y)
    #         for z in (-half.z, half.z)
    #     ]
    #
    #     corners = [rotate_vector(center + offset, center, self.obj.rotation) for offset in offsets]
    #     return corners

    def check_collision(self, other):
        other_collider = getattr(other, 'collider', other)
        if other_collider is None:
            return None

        # --- Internal Functions ---

        def quaternion_to_rotation_matrix(q):
            w, x, y, z = q.w, q.x, q.y, q.z
            xx = x * x
            yy = y * y
            zz = z * z
            xy = x * y
            xz = x * z
            yz = y * z
            wx = w * x
            wy = w * y
            wz = w * z

            return np.array([
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
            ])

        def euler_to_rotation_matrix(euler):
            """
            euler: Vector3D with x=roll, y=pitch, z=yaw, in radians
            """
            cx = math.cos(euler.x)
            sx = math.sin(euler.x)
            cy = math.cos(euler.y)
            sy = math.sin(euler.y)
            cz = math.cos(euler.z)
            sz = math.sin(euler.z)

            R_x = np.array([
                [1, 0, 0],
                [0, cx, -sx],
                [0, sx, cx]
            ])

            R_y = np.array([
                [cy, 0, sy],
                [0, 1, 0],
                [-sy, 0, cy]
            ])

            R_z = np.array([
                [cz, -sz, 0],
                [sz, cz, 0],
                [0, 0, 1]
            ])

            # Combine rotations: R = Rz * Ry * Rx
            R = R_z @ R_y @ R_x
            return R
        def get_axes(rotation):
            # R = euler_to_rotation_matrix(rotation_quaternion)
            #
            # right = Vector3D(*R[:, 0])  # First column
            # up = Vector3D(*R[:, 1])  # Second column
            # forward = Vector3D(*R[:, 2])  # Third column
            right = rotate_vector_old(Vector3D(1, 0, 0), Vector3D(0, 0, 0), rotation).normalized()
            up = rotate_vector_old(Vector3D(0, 1, 0), Vector3D(0, 0, 0), rotation).normalized()
            forward = rotate_vector_old(Vector3D(0, 0, 1), Vector3D(0, 0, 0), rotation).normalized()
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

        # bounding_radius_a = a_half.magnitude()
        # bounding_radius_b = b_half.magnitude()
        # center_distance = (b_center - a_center).magnitude()
        # if center_distance > bounding_radius_a + bounding_radius_b:
        #     return None

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
        if (a_center - b_center).dot(normal) < 0:
            normal = normal * -1
        if self.is_trigger:
            self.OnTriggerEnter(other_collider)
        if other_collider.is_trigger:
            other_collider.OnTriggerEnter(self)
        return contact_point, normal



class Rigidbody:
    def __init__(self, obj=None, mass=1.0, size=Vector3D(1, 1, 1), position=Vector3D(0, 0, 0),
                 center_of_mass=Vector3D(0, 0, 0), velocity=None, angular_velocity=Vector3D(), force=None,
                 isKinematic=False, useGravity=True, drag=1, friction_coefficient=0.6,restitution=0.3):
        self.mass = mass
        self.obj = obj
        self.drag = drag
        self.restitution = restitution
        self.friction_coefficient = friction_coefficient
        self.center_of_mass = center_of_mass if center_of_mass else position
        self.velocity = velocity or Vector3D(0, 0, 0)
        self.acceleration = Vector3D()
        self.angular_acceleration = Vector3D()
        self.torque = Vector3D()
        self.force = force or Vector3D(0, 0, 0)
        self.isKinematic = isKinematic
        self.useGravity = useGravity
        self.angular_velocity = angular_velocity
        self.normal_force = Vector3D()
        if self.obj != None:
            self.inertia = Vector3D(
                (1 / 12) * self.mass * (self.obj.size.y ** 2 + self.obj.size.z ** 2),  # I_x
                (1 / 12) * self.mass * (self.obj.size.x ** 2 + self.obj.size.z ** 2),  # I_y
                (1 / 12) * self.mass * (self.obj.size.x ** 2 + self.obj.size.y ** 2)  # I_z
            )

    def exert_force(self, force):
        # F = ma ⇒ a = F/m
        self.force += force
    def accelerate(self, acceleration, dt=0.01):
        self.acceleration += acceleration
        self.obj.integrat2(dt)

    def apply_torque(self, force: Vector3D, application_point: Vector3D):
        # Offset from center of mass to point of application
        r = application_point - self.obj.position  # self.position = center of mass

        # Torque = r × F (cross product)
        torque = r.cross(force)

        # Add to total torque
        self.torque += torque

    def exert_angular_acceleration(self, force, application_point=Vector3D):
        # Offset from center of mass to point of application
        r = application_point - self.obj.position  # r = lever arm from CoM

        # Torque = r × F
        torque = r.cross(force)

        # Angular acceleration = torque / inertia
        self.angular_acceleration += Vector3D(
            torque.x / self.inertia.x,
            torque.y / self.inertia.y,
            torque.z / self.inertia.z
        )

import PPO
class Joint:
    def __init__(self,other=None, position=None, rotation=None, look_position=True, look_rotation=False):
        self.other = other
        self.look_position = look_position
        self.look_rotation = look_rotation


# class AgentController:
#     def __init__(self, obj, goal, obs_dim=6, action_dim=4):
#         self.obj = obj              # the object this agent controls
#         self.goal = goal            # its target
#         self.agent = PPO.PPOAgent(obs_dim=obs_dim, action_dim=action_dim)
#         self.reset()
#
#     def get_obs(self):
#         return [
#             self.goal.position.x, self.goal.position.y, self.goal.position.z,
#             self.obj.position.x, self.obj.position.y, self.obj.position.z
#         ]
#
#     def act(self):
#         obs = self.get_obs()
#
#         # Get discrete action and value
#         discrete_action, discrete_logprob, value = self.agent.get_discrete_action(obs)
#
#         # Get continuous action
#         continuous_action, continuous_logprob = self.agent.get_continuous_action(obs)
#
#         # Store everything
#         self.last_obs = obs
#         self.last_discrete_action = discrete_action
#         self.last_continuous_action = continuous_action
#         self.last_discrete_logprob = discrete_logprob
#         self.last_continuous_logprob = continuous_logprob
#         self.last_value = value
#
#         # Apply action: pass both
#         self.apply_action(discrete_action, continuous_action)
#
#     def apply_action(self, action):
#         if action == 0:
#             self.obj.rigidbody.velocity.z += 0.5
#         elif action == 1:
#             self.obj.rigidbody.velocity.x += -0.5
#         elif action == 2:
#             self.obj.rigidbody.velocity.z += -0.5
#         elif action == 3:
#             self.obj.rigidbody.velocity.x += 0.5
#
#     def store_experience(self, reward, done):
#         self.agent.store((
#             self.last_obs,
#             self.last_discrete_action,
#             self.last_continuous_action,
#             self.last_discrete_logprob,
#             self.last_continuous_logprob,
#             reward,
#             self.last_value,
#             done
#         ))
#
#     def update(self):
#         self.agent.update()
#
#     def reset(self):
#         self.last_obs = None
#         self.last_action = None
#         self.last_logp = None
#         self.last_value = None

class Object:
    @property
    def quaternion(self):
        if self._rotation_dirty:
            self._quaternion = self._compute_quaternion()
            self._rotation_dirty = False
        return self._quaternion

    def _compute_quaternion(self):
        roll = self.rotation.x
        pitch = self.rotation.y
        yaw = self.rotation.z

        c1 = math.cos(yaw / 2)
        s1 = math.sin(yaw / 2)
        c2 = math.cos(pitch / 2)
        s2 = math.sin(pitch / 2)
        c3 = math.cos(roll / 2)
        s3 = math.sin(roll / 2)

        w = c1 * c2 * c3 + s1 * s2 * s3
        x = c1 * c2 * s3 - s1 * s2 * c3
        y = c1 * s2 * c3 + s1 * c2 * s3
        z = s1 * c2 * c3 - c1 * s2 * s3

        return Vector4(x, y, z, w)
    @property
    def local_position(self):
        if self.parent is None:
            return copy.copy(self.position)

        # Step 1: Offset vector from parent to this object
        offset = self.position - self.parent.position

        # Step 2: Rotate offset into local space (undo parent rotation)
        inverse_quaternion = -self.parent.rotation
        local_offset = rotate_vector_old(offset, Vector3D(0, 0, 0), inverse_quaternion)

        return local_offset
    @local_position.setter
    def local_position(self, new_local_position):
        if self.parent is None:
            self.position = copy.copy(new_local_position)
        else:
            # Step 1: Rotate local position to world space using parent's rotation
            rotated_offset = rotate_vector_old(new_local_position, Vector3D(0, 0, 0), self.parent.rotation)

            # Step 2: Translate by parent position
            self.position = self.parent.position + rotated_offset
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

        obj_copy.__default_position = copy.deepcopy(self.__default_position, memo)

        # Fix parent for children
        obj_copy.components = {}
        for name, comp in self.components.items():
            comp_copy = copy.deepcopy(comp, memo)
            if hasattr(comp_copy, 'obj'):
                comp_copy.obj = obj_copy
            obj_copy.components[name] = comp_copy
            if hasattr(comp_copy, 'parent'):
                comp_copy.parent = obj_copy
        # Fix component references
        # for comp in obj_copy.components.values():
        #     if hasattr(comp, 'obj'):
        #         comp.obj = obj_copy

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

        self.__default_position = copy.copy(self.position)
        # self.__default_position = Vector3D(0,0,0)

        self.rotation = Rotation(*rotation) if isinstance(rotation, tuple) else rotation or Rotation()
        # self.collider = BoxCollider(self.size, self) if size else None
        self.world = None
        # self.local_position = self.local_position()
        self._quaternion = self._compute_quaternion()
        self._rotation_dirty = False


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

    def search_by_component(self, component_name):
        # Check if this object has the desired component
        if hasattr(self, "components") and component_name in self.components:
            return self
        # Recursively check children
        if hasattr(self, "children"):
            for child in self.children:
                result = child.search_by_component(component_name)
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
            rb.inertia = Vector3D(
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
            # joint.other.rigidbody.force = self.rigidbody.force
        elif name == "agent":
            if isinstance(component, tuple):
                agent = PPO.Agent(*component)
            elif isinstance(component, PPO.Agent):
                agent = component
            else:
                raise TypeError("Invalid type for Agent")
        # Save component
        self.components[name] = component
        component.parent = self  # optional back-reference
        if hasattr(component, 'start') and component.start is not None:
            component.start()

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
    # def position(self) -> Vector3D:
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
    #     return Vector3D(rotated[0] + px,
    #                     rotated[1] + py,
    #                     rotated[2] + pz)
    #
    # @position.setter
    # def position(self, value: Vector3D):
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
    #         self.local_position = Vector3D(local_vec[0],
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

    def Stage1(self,children):

        rb = self.get_component("rigidbody")

        center_of_mass = [self.position]
        if rb is not None:
            for child in children:
                center_of_mass.append(child.position)
            self.rigidbody.center_of_mass = Vector3D.mean(center_of_mass)

        if self.parent == None:  # the world does not need an update
            for child in children:
                child.Stage1(children)
            return

        if rb is None or rb.isKinematic:
            return

        # === 2) APPLY GRAVITY (AND TORSOUE DUE TO GRAVITY) ===
        if self.rigidbody.useGravity:
            gravity = Vector3D(0, -9.8, 0)
            self.rigidbody.force += gravity * self.rigidbody.mass

            # If you want torque from gravity:
            r = self.position - self.rigidbody.center_of_mass
            gravity_torque = r.cross(gravity * self.rigidbody.mass)
            self.rigidbody.torque += gravity_torque
    def Stage2(self,children):
        if self.parent == None:  # the world does not need an update
            for child in children:
                child.Stage2(children)
            return
        rb = self.get_component("rigidbody")

        if rb is None or rb.isKinematic:
            return
        joint = self.get_component("joint")
        if joint is not None:
            # Get common acceleration
            # Net force applied to both bodies
            net_force = self.rigidbody.force + self.joint.other.rigidbody.force

            # Common acceleration both bodies should have
            common_acceleration = net_force / (self.rigidbody.mass + self.joint.other.rigidbody.mass)

            # What force each body *should* have to achieve this common acceleration
            desired_force_self = common_acceleration * self.rigidbody.mass
            desired_force_other = common_acceleration * self.joint.other.rigidbody.mass

            # Compute the difference between the actual forces and desired forces
            correction_self = (desired_force_self - self.rigidbody.force)
            correction_other =( desired_force_other - self.joint.other.rigidbody.force)

            # Applying corrections
            self.rigidbody.force += correction_self
            self.joint.other.rigidbody.force += correction_other

    # def Stage3(self,children):
    #
    #
    #     if self.parent == None:  # the world does not need an update
    #         for child in children:
    #             child.Stage3(children)
    #         return
    #     rb = self.get_component("rigidbody")
    #
    #     if rb is None or rb.isKinematic:
    #         return
    #     total_normal_force = Vector3D(0,0,0)
    #     for child in children:
    #         joint = child.get_component('joint')
    #         joint2 = self.get_component('joint')
    #
    #         # if child == self:
    #         if child.get_component("collider") is not None and child != self:#
    #             joint = self.get_component("joint")
    #             # if joint and joint.other == child:
    #             #     continue  # Skip collision with joined body
    #
    #             # if joint and joint.other == self or joint2 and joint2.other == child:
    #             #     continue
    #             result = self.collider.check_collision(child)
    #             if result is None:
    #                 # rb.force *= 0.8
    #                 continue  # No collision
    #             contact_point, normal = result
    #             normal_force = self.rigidbody.force.reduce_vector_along_direction(normal * -1) * -1
    #             normal_force = normal * self.rigidbody.force * -1
    #             total_normal_force += normal_force
    #             if child.get_component("rigidbody") is None or child.rigidbody.isKinematic:
    #                 self.resolve_kinematic_collision(child, normal, contact_point, normal_force)
    #             else:
    #                 self.resolve_dynamic_collision(child, normal, contact_point, normal_force)
    #             # === After updating velocity with all forces ===
    #
    #     if total_normal_force != Vector3D(0,0,0):
    #         self.apply_friction(total_normal_force, dt)
    #     print()

    def Stage3(self, children):

        if self.parent == None:  # the world does not need an update
            for child in children:
                child.Stage3(children)
            return
        rb = self.get_component("rigidbody")

        if rb is None or rb.isKinematic:
            return
        normal_forces = Vector3D()
        for child in children:
            if child == self:
                continue  # Skip self

            if child.get_component("collider") is None:
                continue  # No collider to check against

            result = self.collider.check_collision(child)
            if result is None:
                continue  # No collision

            contact_point, normal = result

            # Project the force onto the collision normal
            force_along_normal = normal * rb.force.dot(normal) * -1

            normal_forces += force_along_normal
            if child.get_component("rigidbody") is None or child.rigidbody.isKinematic:
                self.resolve_kinematic_collision(child, normal, contact_point, force_along_normal)
            else:
                self.resolve_dynamic_collision(child, normal, contact_point, force_along_normal)

        rb.normal_forces = normal_forces

    def Stage4(self, children):

        if self.parent == None:  # the world does not need an update
            for child in children:
                child.Stage4(children)
            return
        rb = self.get_component("rigidbody")

        if rb is None or rb.isKinematic:
            return
        rb.force += rb.normal_forces
        if rb.normal_forces != Vector3D(0, 0, 0):
            self.apply_friction(rb.normal_forces, dt)
        rb.normal_forces = Vector3D()

    def update(self, dt,chack =True):
        children = self.get_all_children_bereshit()
        if chack :
            for child in children:
                for component in child.components.values():
                        if hasattr(component, 'main') and component.main is not None:
                            component.main()

        self.Stage1(children) # APPLY GRAVITY and external forces
        self.Stage2(children) # handel joints
        self.Stage3(children) # handel collins and normal forces
        self.Stage4(children) # handel friction
        for child in children:
            rb = child.get_component("rigidbody")
            if rb is not None:
                child.integrat(dt)

    def resolve_kinematic_collision(self, child, normal, contact_point, normal_force):
        # self.rigidbody.force += normal_force
        if child.get_component("rigidbody") is None:
            e = self.rigidbody.restitution  # Coefficient of restitution (e = 0: perfectly inelastic, e = 1: elastic)
            v2_n = 0  # still allowed, but v2_n should be constant
            v1_n = self.rigidbody.velocity.dot(normal)  # normal component of self's velocity
            relative_velocity = self.rigidbody.velocity - Vector3D(0,0,0)

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



        # self.rigidbody.force += normal_force

        # child.rigidbody.force -= normal_force  # Equal and opposite

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
        friction_limit = normal_force.magnitude() * rb.friction_coefficient

        if rb.velocity.magnitude() > 0:
            # Body is sliding, apply kinetic friction
            d = rb.velocity.normalized() * -1
            friction_acc = d * (friction_limit / rb.mass)
            new_velocity = rb.velocity + friction_acc * dt
            if new_velocity.dot(rb.velocity) <= 0:
                rb.velocity = Vector3D(0, 0, 0)
            else:
                rb.velocity = new_velocity

        elif rb.force.magnitude() > 0:
            # Body is at rest, apply static friction
            d = rb.force.normalized() * -1
            applied_force_magnitude = rb.force.magnitude()
            if applied_force_magnitude <= friction_limit:
                # Static friction cancels all force
                rb.force = Vector3D(0, 0, 0)
            else:
                # Static friction resists up to limit, remainder causes movement
                rb.force += d * friction_limit

    def integrat(self, dt):

        # === 4) INTEGRATION PHASE ===
        # 4.1) Linear acceleration & velocity:
        self.rigidbody.acceleration = self.rigidbody.force / self.rigidbody.mass
        self.rigidbody.acceleration.y *= self.rigidbody.drag

        self.rigidbody.velocity += self.rigidbody.acceleration * dt
        # 4.2) Angular acceleration & velocity (component‐wise):
        self.rigidbody.angular_acceleration = Vector3D(
            self.rigidbody.torque.x / self.rigidbody.inertia.x if self.rigidbody.inertia.x != 0 else 0,
            self.rigidbody.torque.y / self.rigidbody.inertia.y if self.rigidbody.inertia.y != 0 else 0,
            self.rigidbody.torque.z / self.rigidbody.inertia.z if self.rigidbody.inertia.z != 0 else 0
        )
        self.rigidbody.angular_velocity += self.rigidbody.angular_acceleration * dt

        # 4.3) Integrate rotation
        ang_disp = self.rigidbody.angular_velocity * dt \
                   + 0.5 * self.rigidbody.angular_acceleration * dt * dt

        self.position += self.rigidbody.velocity * dt \
                         + 0.5 * self.rigidbody.acceleration * dt * dt
        self.rigidbody.force = Vector3D(0, 0, 0)
        # self.rigidbody.torque = Vector3D(0, 0, 0)
        # self.add_rotation(ang_disp)

        # if self.get_component("joint") != None:
        #     if self.joint.look_position:
        #         self.joint.other.position += self.rigidbody.velocity * dt \
        #                          + 0.5 * self.rigidbody.acceleration * dt * dt
        # 4.5) Angular damping
        # self.rigidbody.angular_velocity *= 0.98

        # 4.6) Reset torques for next frame

        self.rigidbody.angular_acceleration = Vector3D(0, 0, 0)
        self.rigidbody.torque = Vector3D(0, 0, 0)
    # def integrat2(self, dt):
    #
    #     # === 4) INTEGRATION PHASE ===
    #     self.rigidbody.velocity += self.rigidbody.acceleration * dt
    #     # 4.2) Angular acceleration & velocity (component‐wise):
    #     self.rigidbody.angular_acceleration = Vector3D(
    #         self.rigidbody.torque.x / self.rigidbody.inertia.x if self.rigidbody.inertia.x != 0 else 0,
    #         self.rigidbody.torque.y / self.rigidbody.inertia.y if self.rigidbody.inertia.y != 0 else 0,
    #         self.rigidbody.torque.z / self.rigidbody.inertia.z if self.rigidbody.inertia.z != 0 else 0
    #     )
    #     self.rigidbody.angular_velocity += self.rigidbody.angular_acceleration * dt
    #
    #     # 4.3) Integrate rotation
    #     ang_disp = self.rigidbody.angular_velocity * dt \
    #                + 0.5 * self.rigidbody.angular_acceleration * dt * dt
    #     if self.get_component("joint") != None:
    #         if not self.joint.look_position:
    #             # 4.4) Integrate position
    #
    #             self.position += self.rigidbody.velocity * dt \
    #                              + 0.5 * self.rigidbody.acceleration * dt * dt
    #             self.set_position(self.position)
    #         if not self.joint.look_rotation:
    #             self.add_rotation(ang_disp)
    #     else:
    #         # 4.4) Integrate position
    #         self.position += self.rigidbody.velocity * dt \
    #                          + 0.5 * self.rigidbody.acceleration * dt * dt
    #
    #         self.set_position(self.position)
    #
    #         self.add_rotation(ang_disp)
    #
    #     # 4.5) Angular damping
    #     # self.rigidbody.angular_velocity *= 0.98
    #
    #     # 4.6) Reset torques for next frame
    #     self.rigidbody.angular_acceleration = Vector3D(0, 0, 0)
    #     self.rigidbody.torque = Vector3D(0, 0, 0)

    # def set_local_position(self):
    #     for child in self.children:
    #         target = child.obj if isinstance(child, Servo) else child
    #         target.local_position = target.position - target.parent.position
    #         target.set_local_position()

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

    def set_rotation(self, new_world_rot: Vector3D):
        """
        Explicitly set this object's world‐space rotation to `new_world_rot`.
        Also recompute `local_rotation` so that:
            local_rotation = new_world_rot - parent.rotation
        (or = new_world_rot if there's no parent).
        """
        # 1. Assign the new world rotation
        self.rotation = Vector3D(new_world_rot.x, new_world_rot.y, new_world_rot.z)
        self._rotation_dirty = True
        # 2. Compute and store the local offset from parent
        if self.parent is not None:
            self.local_rotation = Vector3D(
                self.rotation.x - self.parent.rotation.x,
                self.rotation.y - self.parent.rotation.y,
                self.rotation.z - self.parent.rotation.z
            )
        else:
            # No parent means local == world
            self.local_rotation = Vector3D(self.rotation.x,
                                           self.rotation.y,
                                           self.rotation.z)

    def add_rotation(self, delta: Vector3D,forall=False):
        """
        Add an incremental rotation `delta` (world‐space Euler angles)
        on top of the current world rotation. Then update local_rotation.
        """
        # 1. Compute the new world rotation by simple vector addition
        new_world_rot = Vector3D(
            self.rotation.x + delta.x,
            self.rotation.y + delta.y,
            self.rotation.z + delta.z
        )

        # 2. Delegate to set_rotation to keep local_rotation in sync
        self.set_rotation(new_world_rot)
        if forall:
            self._set_rotation_recursive(delta)
            self.set_projection(delta)

    def _set_rotation_recursive(self, delta):
        for child in self.children:
            target = child.obj if isinstance(child, Servo) else child
            target.rotation += delta
            target._set_rotation_recursive(delta)

    def update_world_transform(self):
        if self.parent is None:
            self.rotation = self.local_rotation
            self.position = self.local_position
        else:
            self.rotation = self.parent.rotation * self.local_rotation
            rotated_local = rotate_vector_old(self.local_position,Vector3D(0,0,0), self.parent.rotation)
            self.position = self.parent.position + rotated_local

        # Recursively update children
        for child in self.children:
            child.update_world_transform()

    def add_rotation_old(self, rotation):
        self.set_rotation(self.rotation + rotation)

    # def set_rotation_old(self, rotation):
    #     delta = rotation - self.rotation
    #     if delta != Vector3D(0, 0, 0):
    #         self.rotation = rotation
    #         self.local_rotation = rotation - self.parent.rotation
    #         self._set_rotation_recursive(delta)
    #         # self.set_projection()



    def set_projection(self, pivot=np.array([0, 0, 0])):
        for child in self.children:
            target = child.obj if isinstance(child, Servo) else child
            vector = target.position
            # if np.allclose(pivot, [0, 0, 0]):
            pivot = target.parent.position
            angles = target.parent.local_rotation
            rotated = rotate_vector_old(vector, pivot, angles)
            # target.set_position(Vector3D(*rotated))
            target.position = Vector3D(*rotated)
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
    def getdefault_position(self):
        return self.__default_position

    def set_default_position(self):
        self.__default_position = copy.copy(self.position)
        for child in self.children:
            child.set_default_position()

    def reset_to_default(self):
        self.position = copy.copy(self.__default_position)
        if self.get_component("rigidbody") is not None:
            self.rigidbody.acceleration = Vector3D(0,0,0)
            self.rigidbody.velocity = Vector3D(0,0,0)

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
        return (f"{self.name}(\n"
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
            "+z": Vector3D(delta, 0, 0),
            "-z": Vector3D(-delta, 0, 0),
            "+y": Vector3D(0, delta, 0),
            "-y": Vector3D(0, -delta, 0),
            "+x": Vector3D(0, 0, delta),
            "-x": Vector3D(0, 0, -delta),
        }.get(direction, Vector3D(0, 0, 0))

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

def compute_axes_from_euler(rotation):
    """
    Given Euler angles (rotation.x, rotation.y, rotation.z) in degrees,
    compute the rotated basis axes: right, up, forward.

    Returns:
        [right: Vector3D, up: Vector3D, forward: Vector3D]
    """
    # Convert degrees to radians
    angles = np.radians([rotation.x, rotation.y, rotation.z])
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)

    # Build combined rotation matrix: Rz * Ry * Rx
    R = np.array([
        [cy*cz,            -cy*sz,           sy],
        [sx*sy*cz + cx*sz, cx*cz - sx*sy*sz, -sx*cy],
        [sx*sz - cx*sy*cz, sx*cz + cx*sy*sz, cx*cy]
    ])

    # Each column is one axis
    right = Vector3D(*R[:,0])
    up = Vector3D(*R[:,1])
    forward = Vector3D(*R[:,2])

    return [right, up, forward]

def rotate_vector_old(vector, pivot, angles):
    vector = np.array([vector.x, vector.y, vector.z])
    pivot = np.array([pivot.x, pivot.y, pivot.z])
    angles = np.array([angles.x, angles.y, angles.z])

    angle_x, angle_y, angle_z = np.radians(angles)
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(angle_x), -math.sin(angle_x)],
                    [0, math.sin(angle_x), math.cos(angle_x)]])
    R_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-math.sin(angle_y), 0, math.cos(angle_y)]])
    R_z = np.array([[math.cos(angle_z), -math.sin(angle_z), 0],
                    [math.sin(angle_z), math.cos(angle_z), 0],
                    [0, 0, 1]])
    R = R_z @ R_y @ R_x
    temp = R @ (vector - pivot) + pivot

    return Vector3D(temp[0], temp[1], temp[2])
def rotate_vector(vector, pivot, rotation: Vector4):
    """
    Rotate 'vector' around 'pivot' using quaternion 'rotation' (Vector4).
    """
    # Convert to numpy for convenience
    v_np = np.array([vector.x, vector.y, vector.z])
    p_np = np.array([pivot.x, pivot.y, pivot.z])

    # Compute relative position
    rel_v = v_np - p_np

    # Represent as pure quaternion
    v_quat = Vector4(0.0, rel_v[0], rel_v[1], rel_v[2])

    # q * v * q^-1
    q_conj = rotation.conjugate()
    rotated_quat = rotation.multiply(v_quat).multiply(q_conj)

    # Extract rotated vector part
    rotated_v = np.array([rotated_quat.x, rotated_quat.y, rotated_quat.z])

    # Shift back
    result = rotated_v + p_np

    return Vector3D(result[0], result[1], result[2])

# def inverse_rotate_vector(vector, quaternion_angles):
#     inverse_quaternion = Vector3D(-quaternion_angles.x, -quaternion_angles.y, -quaternion_angles.z)
#     return rotate_vector(vector, Vector3D(0, 0, 0), inverse_quaternion)

def euler_to_quaternion(roll, pitch, yaw):
    c1 = math.cos(yaw / 2)
    s1 = math.sin(yaw / 2)
    c2 = math.cos(pitch / 2)
    s2 = math.sin(pitch / 2)
    c3 = math.cos(roll / 2)
    s3 = math.sin(roll / 2)

    w = c1 * c2 * c3 + s1 * s2 * s3
    x = c1 * c2 * s3 - s1 * s2 * c3
    y = c1 * s2 * c3 + s1 * c2 * s3
    z = s1 * c2 * c3 - c1 * s2 * s3

    return (x, y, z, w)

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

def direction_to_angles(v: Vector3D):
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
    return Vector3D(pitch, yaw, 0)  # (pitch, yaw, roll)
