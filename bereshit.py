import math
import time

import moderngl
import numpy as np
from dataclasses import dataclass
import statistics
import copy
import PPO
import send
from math import sqrt
import trimesh
import open3d as o3d
dt = 1/60

@dataclass
class Vector3:

    x: float = 0
    y: float = 0
    z: float = 0

    def floor(self):
        factor = 10 ** 5
        return Vector3(math.floor(self.x * factor) / factor,math.floor(self.y * factor) / factor,math.floor(self.x * factor) / factor)
    def __iadd__(self, other):
        if isinstance(other, Vector3):
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
        return Vector3(-self.x, -self.y, -self.z)
    def __add__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, (list, tuple, np.ndarray)) and len(other) == 3:
            return Vector3(self.x + other[0], self.y + other[1], self.z + other[2])
        raise TypeError(f"Unsupported type for addition: {type(other)}")

    def __sub__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
        raise TypeError("Subtraction only supported between Vector3 instances")

    def __truediv__(self, other):
        if isinstance(other, Vector3):
            if other.x == 0 or other.y == 0 or other.z == 0:
                raise ZeroDivisionError("Division by zero in vector components")

            return Vector3(
                self.x / other.x if other.x != 0 else 0,
                self.y / other.y if other.y != 0 else 0,
                self.z / other.z if other.z != 0 else 0
            )
        elif isinstance(other, (int, float)):
            return Vector3(self.x / other, self.y / other, self.z / other)
        raise TypeError(f"Unsupported division between Vector3 and {type(other)}")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector3(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, Vector3):
            return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
        raise TypeError("Unsupported type for multiplication")
    def __copy__(self):
        return Vector3(self.x,self.y,self.z)

    def __eq__(self, other):
        return isinstance(other, Vector3) and (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __hash__(self):
        return hash((self.x, self.y, self.z))
    def __rmatmul__(self, matrix):
        """
        Enables: matrix @ Vector3
        Assumes matrix is a 3x3 NumPy array or list-of-lists.
        """
        if isinstance(matrix, (list, tuple)):
            matrix = np.array(matrix)
        result = matrix @ self.to_np()
        return Vector3.from_np(result)

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

    def to_np(self):
        return np.array([self.x, self.y, self.z], dtype='f4')

    @classmethod
    def from_np(cls, arr):
        """Create a Vector3 from a NumPy array or list."""
        return cls(arr[0], arr[1], arr[2])

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
class Quaternion:
    def __init__(self, x=0, y=0, z=0, w=1):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __repr__(self):
        return f"Quaternion({self.x}, {self.y}, {self.z}, {self.w})"

    def __copy__(self):
        return Quaternion(self.x, self.y, self.z, self.w)

    def __neg__(self):
        return Quaternion(-self.x, -self.y, -self.z, -self.w)

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)
        elif isinstance(other, (list, tuple)) and len(other) == 4:
            return Quaternion(self.x + other[0], self.y + other[1], self.z + other[2], self.w + other[3])
        raise TypeError(f"Unsupported type for addition: {type(other)}")

    def __iadd__(self, other):
        result = self + other
        self.x, self.y, self.z, self.w = result.x, result.y, result.z, result.w
        return self

    def __sub__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)
        elif isinstance(other, (list, tuple)) and len(other) == 4:
            return Quaternion(self.x - other[0], self.y - other[1], self.z - other[2], self.w - other[3])
        raise TypeError(f"Unsupported type for subtraction: {type(other)}")

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            # Hamilton product
            return Quaternion(
                self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
                self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
                self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
                self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            )
        raise TypeError("Quaternion can only be multiplied by another Quaternion")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(self.x / other, self.y / other, self.z / other, self.w / other)
        elif isinstance(other, Quaternion):
            return self * other.inverse()
        raise TypeError("Unsupported type for division")

    def conjugate(self):
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    def norm(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)

    def normalized(self):
        n = self.norm()
        if n == 0:
            return Quaternion(0, 0, 0, 1)
        return self / n

    def inverse(self):
        norm_sq = self.x ** 2 + self.y ** 2 + self.z ** 2 + self.w ** 2
        return Quaternion(
            -self.x / norm_sq,
            -self.y / norm_sq,
            -self.z / norm_sq,
            self.w / norm_sq
        )

    def to_euler(self):
        """
        Converts the quaternion to Euler angles (roll, pitch, yaw) in radians.
        Convention: ZYX (yaw-pitch-roll)
        Returns:
            (roll, pitch, yaw)
        """
        x, y, z, w = self.x, self.y, self.z, self.w

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return Vector3(math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

    def euler(self):
        roll = self.x
        pitch = self.y
        yaw = self.z

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

        return Quaternion(x, y, z, w)

    @staticmethod
    def axis_angle(axis, angle_rad):
        """Create a quaternion representing a rotation of angle_rad around 'axis' (Vector3)."""
        half_angle = angle_rad * 0.5
        sin_half = math.sin(half_angle)
        axis_n = axis.normalized()  # Make sure your Vector3 has this method

        return Quaternion(
            axis_n.x * sin_half,
            axis_n.y * sin_half,
            axis_n.z * sin_half,
            math.cos(half_angle)
        )

    def to_axis_angle(self):
        """Convert this quaternion to (axis: Vector3, angle: float)"""
        if abs(self.w) > 1:
            self = self.normalized()

        angle = 2 * math.acos(self.w)
        s = math.sqrt(1 - self.w * self.w)

        if s < 1e-6:
            # If s is too small, return arbitrary axis
            return Vector3(1, 0, 0), 0.0
        else:
            return Vector3(self.x / s, self.y / s, self.z / s), angle

    def to_matrix3(self):
        """Returns a 3x3 rotation matrix (as numpy array) from this quaternion."""
        x, y, z, w = self.x, self.y, self.z, self.w

        return np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
        ])
    def rotate(self, v: Vector3) -> Vector3:
        """
        Rotate a Vector3 `v` by this quaternion.
        """
        qv = Quaternion(v.x, v.y, v.z, 0)
        rotated = self * qv * self.inverse()
        return Vector3(rotated.x, rotated.y, rotated.z)
class Mesh_rander:
    def attach(self, owner_object):
        return "Mesh"
    def __init__(self, vertices=None, edges=None, shape=None, obj_path=None):
        self.shape = shape
        self.colors = None
        self.ctx = moderngl.create_standalone_context()
        if obj_path:
            self.vertices, self.triangles, self.edges, self.colors, self.vertex_shader, self.fragment_shader = self.load_model(
                obj_path)

            # Compile program
            self.prog = self.ctx.program(
                vertex_shader=self.vertex_shader,
                fragment_shader=self.fragment_shader,
            )

        elif shape and vertices is None and edges is None:
            generator = self.get_generator_function(shape)
            if generator:
                result = generator()
                if len(result) == 2:  # Only vertices and edges
                    self.vertices, self.edges = result
                    self.triangles = None  # No triangle data available
                elif len(result) == 3:  # Vertices, edges, triangles
                    self.vertices, self.edges, self.triangles = result
                else:
                    raise ValueError("Shape generator must return (vertices, edges) or (vertices, edges, triangles)")
            else:
                raise ValueError(f"No generator found for shape: {shape}")

        elif vertices is not None and edges is not None:
            self.vertices = vertices
            self.edges = edges
            self.triangles = None  # User didn’t provide triangles

        else:
            raise ValueError("Must provide either a shape, .obj path, or both vertices and edges")

    def get_generator_function(self, shape_name):
        generators = {
            "box": self.generate_cube,
            "ellipsoid": self.generate_ellipsoid,
            "cone": self.generate_cone,
            "cylinder": self.generate_cylinder,
            "pyramid": self.generate_pyramid,
            "triangular_prism": self.generate_triangular_prism,
            "empty": self.generate_empty

        }
        return generators.get(shape_name)

    @staticmethod
    def load_model(path):
        mesh = trimesh.load(path, force='mesh')

        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Loaded file is not a mesh")

        # Convert vertices
        # vertices = [Vector3(*v) for v in mesh.vertices]
        # Convert and center vertices
        vertices = [Vector3(*v) for v in mesh.vertices]
        centroid = sum(vertices, Vector3()) * (1.0 / len(vertices))
        vertices = [v - centroid for v in vertices]

        # Convert faces to triangles (used for solid rendering)
        triangles = [tuple(face) for face in mesh.faces]

        # Convert faces to edges (used for wireframe rendering)
        edge_set = set()
        for face in mesh.faces:
            for i in range(3):
                a = face[i]
                b = face[(i + 1) % 3]
                edge_set.add(tuple(sorted((a, b))))
        edges = list(edge_set)

        # Load vertex colors if available
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            raw_colors = mesh.visual.vertex_colors[:, :3]
            colors = [tuple(c / 255.0) for c in raw_colors]
        elif hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
            diffuse = mesh.visual.material.diffuse
            if diffuse is not None:
                colors = [tuple(diffuse)] * len(vertices)
            else:
                colors = [(1.0, 1.0, 1.0)] * len(vertices)  # default white
        else:
            colors = [(1.0, 1.0, 1.0)] * len(vertices)  # default white

        # Vertex Shader
        vertex_shader = """
        #version 330

        in vec3 in_position;
        in vec3 in_color;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        out vec3 v_color;

        void main() {
            gl_Position = projection * view * model * vec4(in_position, 1.0);
            v_color = in_color;
        }
        """

        # Fragment Shader
        fragment_shader = """
        #version 330

        in vec3 v_color;
        out vec4 fragColor;

        void main() {
            fragColor = vec4(v_color, 1.0);
        }
        """

        return vertices, triangles, edges, colors, vertex_shader, fragment_shader

    @staticmethod
    def generate_cube():
        cube_vertices = [
            Vector3(-1, -1, -1), Vector3(1, -1, -1),
            Vector3(1, 1, -1), Vector3(-1, 1, -1),
            Vector3(-1, -1, 1), Vector3(1, -1, 1),
            Vector3(1, 1, 1), Vector3(-1, 1, 1),
        ]
        cube_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        # centroid = sum(cube_vertices, Vector3()) * (1.0 / len(cube_vertices))
        # cube_vertices = [v - centroid for v in cube_vertices]
        return cube_vertices, cube_edges
    @staticmethod
    def generate_ellipsoid(rx=1, ry=1, rz=1, segments=12, rings=6):
        vertices = []
        edges = []

        for i in range(rings + 1):
            phi = math.pi * i / rings
            for j in range(segments):
                theta = 2 * math.pi * j / segments
                x = rx * math.sin(phi) * math.cos(theta)
                y = ry * math.cos(phi)
                z = rz * math.sin(phi) * math.sin(theta)
                vertices.append(Vector3(x, y, z))

        for i in range(rings):
            for j in range(segments):
                current = i * segments + j
                next_seg = current + 1 if (j + 1) < segments else i * segments
                next_ring = current + segments
                edges.append((current, next_seg))
                if i < rings:
                    edges.append((current, next_ring))
        return vertices, edges
    @staticmethod
    def generate_cone(radius=1, height=2, segments=12):
        vertices = [Vector3(0, height / 2, 0)]  # Tip of cone
        base_center = Vector3(0, -height / 2, 0)
        base_indices = []

        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            vertices.append(Vector3(x, -height / 2, z))
            base_indices.append(i + 1)

        edges = [(0, i) for i in base_indices]  # Sides to tip
        for i in range(segments):
            edges.append((base_indices[i], base_indices[(i + 1) % segments]))  # Base circle
        return vertices, edges
    @staticmethod
    def generate_pyramid(base_size=2, height=2):
        half = base_size / 2
        vertices = [
            Vector3(-half, 0, -half),  # 0 base
            Vector3(half, 0, -half),  # 1
            Vector3(half, 0, half),  # 2
            Vector3(-half, 0, half),  # 3
            Vector3(0, height, 0)  # 4 tip
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # base
            (0, 4), (1, 4), (2, 4), (3, 4)  # sides
        ]
        return vertices, edges
    @staticmethod
    def generate_cylinder(radius=1, height=2, segments=16):
        vertices = []
        edges = []

        # Generate bottom and top circle vertices
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            vertices.append(Vector3(x, -height / 2, z))  # Bottom circle
            vertices.append(Vector3(x, height / 2, z))  # Top circle

        for i in range(segments):
            bottom = i * 2
            top = bottom + 1
            next_bottom = (bottom + 2) % (2 * segments)
            next_top = (top + 2) % (2 * segments)

            # Vertical edge
            edges.append((bottom, top))

            # Bottom circle edge
            edges.append((bottom, next_bottom))

            # Top circle edge
            edges.append((top, next_top))

        return vertices, edges

    @staticmethod
    def generate_triangular_prism(base=1, height=1, depth=2):
        h = math.sqrt(3) * base / 2
        vertices = [
            Vector3(-base / 2, -h / 3, -depth / 2),  # bottom front triangle
            Vector3(base / 2, -h / 3, -depth / 2),
            Vector3(0, 2 * h / 3, -depth / 2),
            Vector3(-base / 2, -h / 3, depth / 2),  # back triangle
            Vector3(base / 2, -h / 3, depth / 2),
            Vector3(0, 2 * h / 3, depth / 2)
        ]
        edges = [
            (0, 1), (1, 2), (2, 0),  # front
            (3, 4), (4, 5), (5, 3),  # back
            (0, 3), (1, 4), (2, 5)  # connecting edges
        ]
        return vertices, edges

    @staticmethod
    def generate_empty():
        cube_vertices = []
        cube_edges = []
        return cube_vertices, cube_edges

class Camera:
    def __init__(self,width = 1920, hight =1080, FOV = 120,VIEWER_DISTANCE = 0, shading="wire"):
        self.width = width
        self.hight = hight
        self.FOV = FOV
        self.VIEWER_DISTANCE = VIEWER_DISTANCE
        self.shading = shading



class Position(Vector3): pass


class LocalPosition(Vector3): pass


class CenterOfGravity(Vector3): pass


class Rotation(Vector3): pass


class LocalRotation(Vector3): pass


class Size(Vector3):
    def __init__(self, x=1, y=1, z=1):
        super().__init__(x, y, z)

class MeshCollider:
    def __init__(self, vertices=None, object_pointer=None, is_trigger=False):
        self.vertices = copy.deepcopy(vertices) or []  # Local-space Vector3
        self.obj = object_pointer
        self.is_trigger = is_trigger
        self.enter = False

    def generate_convex_hull_and_simplify(self, target_triangles=500):
        """
        Simplify a convex hull built from the given vertices using Open3D.
        """
        points_np = np.array([v.to_tuple() for v in self.vertices])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)

        hull, _ = pcd.compute_convex_hull()
        simplified = hull.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
        self.vertices = [Vector3(*p) for p in np.asarray(simplified.vertices)]

    def get_world_vertices(self):
        return [
            rotate_vector_old(v, self.obj.position, self.obj.rotation) + self.obj.position
            for v in self.vertices
        ]

    def extract_triangle_edges(self, vertices):
        edges = set()
        for i in range(0, len(vertices), 3):
            if i + 2 < len(vertices):
                a, b, c = vertices[i:i+3]
                edges.update([
                    (a, b),
                    (b, c),
                    (c, a),
                ])
        return edges

    def get_axes(self, verts_a_world, verts_b_world):
        axes = set()

        def extract_normals(vertices):
            for i in range(0, len(vertices), 3):
                if i + 2 < len(vertices):
                    a, b, c = vertices[i:i+3]
                    normal = (b - a).cross(c - a).normalized()
                    if normal.magnitude() > 1e-6:
                        axes.add(normal.to_tuple())

        # Face normals
        extract_normals(verts_a_world)
        extract_normals(verts_b_world)

        # Edge cross products
        edges_a = self.extract_triangle_edges(verts_a_world)
        edges_b = self.extract_triangle_edges(verts_b_world)

        for ea in edges_a:
            dir_a = (ea[1] - ea[0]).normalized()
            for eb in edges_b:
                dir_b = (eb[1] - eb[0]).normalized()
                cross = dir_a.cross(dir_b)
                if cross.magnitude() > 1e-6:
                    axes.add(cross.normalized().to_tuple())

        return list(axes)

    def project(self, vertices, axis):
        axis = Vector3(*axis) if isinstance(axis, tuple) else axis
        dots = [v.dot(axis) for v in vertices]
        return min(dots), max(dots)

    def check_collision(self, other):
        other_collider = getattr(other, 'collider', other)
        if other_collider is None:
            return None

        verts_a_world = self.get_world_vertices()
        verts_b_world = other_collider.get_world_vertices()

        axes = self.get_axes(verts_a_world, verts_b_world)

        smallest_overlap = float('inf')
        collision_axis = None

        for axis in axes:
            proj1 = self.project(verts_a_world, axis)
            proj2 = self.project(verts_b_world, axis)

            if proj1[1] < proj2[0] or proj2[1] < proj1[0]:
                return None  # Separating axis found

            overlap = min(proj1[1], proj2[1]) - max(proj1[0], proj2[0])
            if overlap < smallest_overlap:
                smallest_overlap = overlap
                collision_axis = axis

        # Compute collision normal and contact point
        normal = Vector3(*collision_axis)
        center_a = sum(verts_a_world, Vector3(0, 0, 0)) * (1 / len(verts_a_world))
        center_b = sum(verts_b_world, Vector3(0, 0, 0)) * (1 / len(verts_b_world))

        if (center_a - center_b).dot(normal) < 0:
            normal = normal * -1

        contact_point = (center_a + center_b) * 0.5
        return contact_point, normal, smallest_overlap

    def attach(self, owner_object):
        self.size = owner_object.size
        self.obj = owner_object
        self.vertices = owner_object.mesh.vertices
        self.generate_convex_hull_and_simplify()
        return "collider"

class SphereCollider:
    def __init__(self, radii=Vector3(1, 1, 1), object_pointer=None, is_trigger=False):
        self.radii = radii
        self.obj = object_pointer
        self.is_trigger = is_trigger
        self.enter = False

    def attach(self, owner_object):
        self.obj = owner_object
        self.radii = owner_object.size * 0.5  # assuming size represents full diameter

        return "collider"
    def OnCollisionEnter(self, other_collider):
        self.enter = True
        for component in self.parent.components.values():
            if hasattr(component, 'OnCollisionEnter') and component.OnCollisionEnter is not None and component != self:
                component.OnCollisionEnter(other_collider)

    def OnCollisionStay(self, other_collider):
        for component in self.parent.components.values():
            if hasattr(component, 'OnCollisionStay') and component.OnCollisionStay is not None and component != self:
                component.OnCollisionStay(other_collider)

    def OnCollisionExit(self, other_collider):
        self.enter = False
        for component in self.parent.components.values():
            if hasattr(component, 'OnCollisionExit') and component.OnCollisionExit is not None and component != self:
                component.OnCollisionExit(other_collider)

    def OnTriggerEnter(self, other_collider):
        for component in self.parent.components.values():
            if hasattr(component, 'OnTriggerEnter') and component.OnTriggerEnter is not None and component != self:
                component.OnTriggerEnter(other_collider)

    def check_collision(self, other):
        other_collider = getattr(other, 'collider', other)
        if other_collider is None:
            return None

        a_center = self.obj.position
        b_center = other_collider.obj.position

        # Convert both ellipsoids into unit spheres (by scaling)
        a_to_unit = (b_center - a_center) / self.radii
        b_to_unit = Vector3(0, 0, 0)
        if isinstance(other_collider, SphereCollider):
            b_to_unit = (a_center - b_center) / other_collider.radii
        else:
            # assume box collider is symmetric cube with uniform radius
            box_half = other_collider.size * 0.5
            b_to_unit = (a_center - b_center) / box_half

        dist = a_to_unit.magnitude()
        if dist < 2:  # in unit sphere space, two radii = 1 + 1
            contact_point = (a_center + b_center) * 0.5
            normal = (a_center - b_center).normalized()
            overlap = 2 - dist  # estimated penetration depth

            if self.is_trigger:
                self.OnTriggerEnter(other_collider)
            if other_collider.is_trigger:
                other_collider.OnTriggerEnter(self)

            if not self.enter:
                self.OnCollisionEnter(other_collider)
            else:
                self.OnCollisionStay(other_collider)

            if not other_collider.enter:
                other_collider.OnCollisionEnter(self)
            else:
                other_collider.OnCollisionStay(self)

            return contact_point, normal, overlap

        # If previously in contact but no longer colliding
        if self.enter:
            self.OnCollisionExit(other_collider)
        if other_collider.enter:
            other_collider.OnCollisionExit(self)

        return None

class BoxCollider:
    def __init__(self, size=Vector3(1, 1, 1), object_pointer=None,is_trigger = False):
        self.size = size
        self.obj = object_pointer
        self.is_trigger = is_trigger
        self.enter = False

    def OnCollisionEnter(self, other_collider):
        self.enter = True

        for component in self.parent.components.values():
            if hasattr(component, 'OnCollisionEnter') and component.OnCollisionEnter is not None and component != self:
                component.OnCollisionEnter(other_collider)
    def OnCollisionStay(self, other_collider):
        for component in self.parent.components.values():
            if hasattr(component, 'OnCollisionStay') and component.OnCollisionStay is not None and component != self:
                component.OnCollisionStay(other_collider)
    def OnCollisionExit(self, other_collider):
        self.enter = False
        for component in self.parent.components.values():
            if hasattr(component, 'OnCollisionExit') and component.OnCollisionExit is not None and component != self:
                component.OnCollisionExit(other_collider)
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
        world_corners = [rotate_vector_old(corner, self.obj.position, self.obj.rotation) + self.obj.position for corner in
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


    def check_collision(self, other):
        other_collider = getattr(other, 'collider', other)
        if other_collider is None:
            return None

        # --- Internal Functions ---

        def get_axes(rotation):
            R = get_rotation_matrix(rotation)

            right_v = R @ np.array([1, 0, 0])
            up_v = R @ np.array([0, 1, 0])
            forward_v = R @ np.array([0, 0, 1])

            right = Vector3(*right_v).normalized()
            up = Vector3(*up_v).normalized()
            forward = Vector3(*forward_v).normalized()
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


                # if self.enter:
                #     self.OnCollisionExit(other_collider)
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

        if self.enter == False:
            self.OnCollisionEnter(other_collider)
        else:
            self.OnCollisionStay(other_collider)

        if other_collider.enter == False:
            other_collider.OnCollisionEnter(self)
        else:
            other_collider.OnCollisionStay(self)
        return contact_point, normal, smallest_overlap

    def attach(self, owner_object):
        self.size = owner_object.size
        self.obj = owner_object
        return "collider"

class Material:
    COLOR_MAP = {
        "white": (1.0, 1.0, 1.0),
        "black": (0.0, 0.0, 0.0),
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "gray": (0.5, 0.5, 0.5),
        # Add more as needed
    }

    def __init__(self, kind="Steel", color="white"):
        self.kind = kind
        # self.color = color
        if isinstance(color, tuple) and len(color) == 3 and all(isinstance(c, (int, float)) for c in color):
            # Already RGB
            self.color = color
        else:
            self.color = self.COLOR_MAP.get(color.lower(), (1.0, 1.0, 1.0))  # default to white
    def attach(self,owner_object):
        return "material"
class Rigidbody:
    _friction_table = {
        ("Steel", "Concrete"): 0.6,
        ("Rubber", "Concrete"): 0.9,
        ("Rubber", "Steel"): 0.8,
        ("Wood", "Ice"): 0.04,
        ("Steel", "Steel"): 0.5,
        ("floor", "Steel"): 0.2,
        ("Steel", "Steel"): 0.2,
        # ("Steel", "Ice"): 0.1,
        # add more as needed
    }
    _default_friction = 0.6
    def __init__(self, obj=None, mass=1.0, size=Vector3(1, 1, 1), position=Vector3(0, 0, 0),
                 center_of_mass=Vector3(0, 0, 0), velocity=None, angular_velocity=None, force=None,
                 isKinematic=False, useGravity=True, drag=0.8, friction_coefficient=0.0,restitution=0.7):
        self.mass = mass
        self.material = ""
        self.drag = drag
        self.obj = obj
        self.restitution = restitution
        self.friction_coefficient = friction_coefficient if friction_coefficient is not None else self._default_friction
        self.center_of_mass = center_of_mass if center_of_mass else position
        self.velocity = velocity or Vector3(0, 0, 0)
        self.acceleration = Vector3()
        self.angular_acceleration = Vector3(0,0,0)
        self.torque = Vector3()
        self.force = force or Vector3(0, 0, 0)
        self.isKinematic = isKinematic

        self.useGravity = useGravity

        if angular_velocity is None:
            angular_velocity = Vector3(0, 0, 0)
        self.angular_velocity = angular_velocity

        self.normal_force = Vector3()

    def _get_friction(self, other_rb):
        """
        Returns the friction coefficient for the pair of materials.
        """
        if not other_rb:
            return self.friction_coefficient

        mat1 = self.material
        mat2 = other_rb.material
        key = (mat1, mat2)
        rev_key = (mat2, mat1)

        if key in Rigidbody._friction_table:
            return Rigidbody._friction_table[key]
        elif rev_key in Rigidbody._friction_table:
            return Rigidbody._friction_table[rev_key]
        else:
            return Rigidbody._default_friction
    def exert_force(self, force):
        # F = ma ⇒ a = F/m
        self.force += force
    def accelerate(self, acceleration, dt=0.01):
        self.acceleration += acceleration
        self.obj.integrat2(dt)

    # def apply_torque(self, force: Vector3, application_point: Vector3):
    #     # Offset from center of mass to point of application
    #     r = application_point - self.obj.position  # self.position = center of mass
    #
    #     # Torque = r × F (cross product)
    #     torque = r.cross(force)
    #
    #     # Add to total torque
    #     self.torque += torque

    # def exert_angular_acceleration(self, force, application_point=Vector3):
    #     # Offset from center of mass to point of application
    #     r = application_point - self.obj.position  # r = lever arm from CoM
    #
    #     # Torque = r × F
    #     torque = r.cross(force)
    #
    #     # Angular acceleration = torque / inertia
    #     self.angular_acceleration += Vector3(
    #         torque.x / self.inertia.x,
    #         torque.y / self.inertia.y,
    #         torque.z / self.inertia.z
    #     )

    def attach(self, owner_object):
        self.size = owner_object.size
        self.position = owner_object.position
        self.center_of_mass = owner_object.position
        self.obj = owner_object
        self.material = owner_object.material.kind

        EPSILON = 1e-8  # Small value to avoid division by zero

        self.inertia = Vector3(
            (1 / 12) * self.mass * (owner_object.size.y ** 2 + owner_object.size.z ** 2),  # I_x
            (1 / 12) * self.mass * (owner_object.size.x ** 2 + owner_object.size.z ** 2),  # I_y
            (1 / 12) * self.mass * (owner_object.size.x ** 2 + owner_object.size.y ** 2)  # I_z
        )

        def safe_inverse(value):
            return 1.0 / value if abs(value) > EPSILON else 0.0

        self.inverse_inertia = np.diag([
            safe_inverse(self.inertia.x),
            safe_inverse(self.inertia.y),
            safe_inverse(self.inertia.z)
        ])


class Joint:
    def __init__(self,other=None, position=None, rotation=None, look_position=True, look_rotation=False):
        self.other = other
        self.look_position = look_position
        self.look_rotation = look_rotation

class FixJoint:
    def __init__(self, other_object):
        """
        other_object: the Object you want to fix to.
        """
        self.other_object = other_object
        self.bodyA = None  # Will be filled in at attach time
        self.bodyB = other_object.get_component("Rigidbody")

        # Compute initial offset
        self.local_offset = None

    def attach(self, owner_object):
        """
        Called when this component is attached to an object.
        """
        self.bodyA = owner_object.get_component("Rigidbody")
        if self.bodyA is None or self.bodyB is None:
            raise ValueError("FixJoint requires both objects to have rigidbodies")
        if self.bodyB.isKinematic:
            raise ValueError("can not joint a Kinematic body")
        self.local_offset = self.bodyB.position - self.bodyA.position
        self.anchor_world = self.bodyA.position + self.local_offset
        self.defaultA = self.bodyA.parent.quaternion
        self.defaultB = self.bodyB.parent.quaternion
        return  "joint"

    def solve(self, dt):
        """
        Enforce linear velocity matching at the joint point (no relative motion).
        Only linear impulse correction (ignores angular).
        """
        deltaA = self.bodyA.parent.quaternion.conjugate() * self.defaultA

        deltaB = self.defaultB - self.bodyB.parent.quaternion
        local_offset = rotate_vector_quaternion(self.local_offset,deltaA)


        # Velocities at anchor points
        vA = self.bodyA.velocity  # ignoring angular contribution
        vB = self.bodyB.velocity

        # Relative velocity
        v_rel = vB - vA

        # Effective mass
        inv_massA = 1.0 / self.bodyA.mass if self.bodyA.mass > 0 else 0.0
        inv_massB = 1.0 / self.bodyB.mass if self.bodyB.mass > 0 else 0.0
        if self.bodyA.isKinematic:
            inv_massA = 0
        effective_mass = 1.0 / (inv_massA + inv_massB) if (inv_massA + inv_massB) > 0 else 0.0

        # Compute impulse to cancel relative velocity
        impulse = v_rel * (-effective_mass)
        target_position_B = self.bodyA.position + local_offset
        correction = target_position_B - self.bodyB.position
        if not self.bodyA.isKinematic and not self.bodyB.isKinematic:
            # Both dynamic — apply impulse to both
            self.bodyA.velocity -= impulse * inv_massA
            self.bodyB.velocity += impulse * inv_massB
        elif not self.bodyB.isKinematic:
            # Only B is dynamic — treat A as fixed
            self.bodyB.velocity += impulse * inv_massB
            self.bodyB.position += correction



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
#             self.obj.Rigidbody.velocity.z += 0.5
#         elif action == 1:
#             self.obj.Rigidbody.velocity.x += -0.5
#         elif action == 2:
#             self.obj.Rigidbody.velocity.z += -0.5
#         elif action == 3:
#             self.obj.Rigidbody.velocity.x += 0.5
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


    def _compute_quaternion(self):
        roll = math.radians(self.rotation.x)
        pitch = math.radians(self.rotation.y)
        yaw = math.radians(self.rotation.z)

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

        return Quaternion(x, y, z, w)
    @property
    def local_position(self):
        if self.parent is None:
            return copy.copy(self.position)

        # Step 1: Offset vector from parent to this object
        offset = self.position - self.parent.position

        # Step 2: Rotate offset into local space (undo parent rotation)
        inverse_quaternion = -self.parent.rotation
        local_offset = rotate_vector_old(offset, Vector3(0, 0, 0), inverse_quaternion)

        return local_offset
    @local_position.setter
    def local_position(self, new_local_position):
        if self.parent is None:
            self.position = copy.copy(new_local_position)
        else:
            # Step 1: Rotate local position to world space using parent's rotation
            rotated_offset = rotate_vector_old(new_local_position, Vector3(0, 0, 0), self.parent.rotation)

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

    def add_component(self, component, name=None):
        if name is None:
            name = component.__class__.__name__
            if hasattr(component, "attach"):
                result = component.attach(self)
                if result is not None:
                    name = result
        else:
            if hasattr(component, "attach"):
                component.attach(self)  # call it, but ignore the result

        self.components[name] = component
        component.parent = self  # optional back-reference
        if hasattr(component, 'start') and component.start is not None:
            component.start()

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
        # self.__default_position = Vector3(0,0,0)

        self.rotation = Rotation(*rotation) if isinstance(rotation, tuple) else rotation or Rotation()
        # self.collider = BoxCollider(self.size, self) if size else None
        self.world = None
        # self.local_position = self.local_position()
        self.quaternion = self._compute_quaternion()
        self._rotation_dirty = False
        self.add_component(Material())
        self.add_component(Mesh_rander(shape="box"))


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

    def Stage1(self,children):

        rb = self.get_component("Rigidbody")

        center_of_mass = [self.position]
        if rb is not None:
            for child in children:
                center_of_mass.append(child.position)
            self.Rigidbody.center_of_mass = Vector3.mean(center_of_mass)

        if self.parent == None:  # the world does not need an update
            for child in children:
                child.Stage1(children)
            return

        if rb is None or rb.isKinematic:
            return

        # === 2) APPLY GRAVITY (AND TORSOUE DUE TO GRAVITY) ===
        if self.Rigidbody.useGravity:
            gravity = Vector3(0, -9.8, 0)
            self.Rigidbody.force += gravity * self.Rigidbody.mass

            # If you want torque from gravity:
            # r = self.position - self.Rigidbody.center_of_mass
            # gravity_torque = r.cross(gravity * self.Rigidbody.mass)
            # self.Rigidbody.torque += gravity_torque
    def Stage2(self,children):
        if self.parent == None:  # the world does not need an update
            for child in children:
                child.Stage2(children)
            return
        rb = self.get_component("Rigidbody")

        if rb is None or rb.isKinematic:
            return
        joint = self.get_component("joint")
        if joint is not None:
            # Get common acceleration
            # Net force applied to both bodies
            net_force = self.Rigidbody.force + self.joint.other.Rigidbody.force

            # Common acceleration both bodies should have
            common_acceleration = net_force / (self.Rigidbody.mass + self.joint.other.Rigidbody.mass)

            # What force each body *should* have to achieve this common acceleration
            desired_force_self = common_acceleration * self.Rigidbody.mass
            desired_force_other = common_acceleration * self.joint.other.Rigidbody.mass

            # Compute the difference between the actual forces and desired forces
            correction_self = (desired_force_self - self.Rigidbody.force)
            correction_other =( desired_force_other - self.joint.other.Rigidbody.force)

            # Applying corrections
            self.Rigidbody.force += correction_self
            self.joint.other.Rigidbody.force += correction_other

    def Stage3(self, children, dt):
        contacts = []

        beta = 0.05# softness factor for positional correction

        # STEP 1: Collect all contacts
        for i in range(len(children)):
            obj1 = children[i]
            rb1 = obj1.get_component("Rigidbody")

            for j in range(i + 1, len(children)):
                obj2 = children[j]
                rb2 = obj2.get_component("Rigidbody")

                # Skip if neither has a Rigidbody or both are kinematic
                if (rb1 is None or rb1.isKinematic) and (rb2 is None or rb2.isKinematic):
                    continue

                result = obj1.collider.check_collision(obj2)
                if result is None:
                    continue

                contact_point, normal, penetration = result

                v1 = rb1.velocity if rb1 and not rb1.isKinematic else Vector3(0, 0, 0)
                v2 = rb2.velocity if rb2 and not rb2.isKinematic else Vector3(0, 0, 0)
                v_rel = v1 - v2
                v_norm = v_rel.dot(normal)

                contacts.append({
                    "rb1": rb1,
                    "rb2": rb2,
                    "normal": normal,
                    "v_norm": v_norm,
                    "penetration": penetration,
                    "contact_point": contact_point
                })

        N = len(contacts)
        if N == 0:
            return

        # STEP 2: Build matrix A
        A = np.zeros((N, N))
        for i, ci in enumerate(contacts):
            ni = ci["normal"]
            rb1i = ci["rb1"]
            rb2i = ci["rb2"]

            for j, cj in enumerate(contacts):
                nj = cj["normal"]
                rb1j = cj["rb1"]
                rb2j = cj["rb2"]

                term = 0.0
                if rb1i is not None and not rb1i.isKinematic:
                    if rb1i == rb1j or rb1i == rb2j:
                        term += ni.dot(nj) / rb1i.mass
                if rb2i is not None and not rb2i.isKinematic:
                    if rb2i == rb1j or rb2i == rb2j:
                        term += ni.dot(nj) / rb2i.mass
                A[i, j] = term

        # STEP 3: Build RHS vector b
        b = np.zeros(N)
        for i, c in enumerate(contacts):
            rb1 = c["rb1"]
            rb2 = c["rb2"]

            # Default restitution
            restitution = 0.0
            if rb1 and rb2:
                restitution = min(rb1.restitution, rb2.restitution)
            elif rb1:
                restitution = rb1.restitution
            elif rb2:
                restitution = rb2.restitution
            if c["penetration"] > 0:
                b[i] = -restitution * c["v_norm"] + beta * c["penetration"] / dt
            else:
                b[i] = -restitution * c["v_norm"]

        # STEP 4: Solve impulses
        impulses = np.linalg.pinv(A) @ b

        impulses = np.maximum(impulses, 0.0)

        # STEP 5: Apply impulses using helper functions
        for i, contact in enumerate(contacts):
            J = impulses[i]
            if contact["v_norm"] > 0:
                continue
            print(f"[DEBUG] v_norm = {contact['v_norm']:.5f}, J = {J:.5f}")

            if contact["rb1"] and contact["rb2"]:
                if not contact["rb1"].isKinematic and not contact["rb2"].isKinematic: # contact["rb1"].parent.name == "obj" and contact["rb2"].parent.name == "obj2"
                    self.resolve_dynamic_collision(contact, J)
                    self.apply_friction_impulse(contact, normal, J)
                elif not contact["rb1"].isKinematic or not contact["rb2"].isKinematic:
                    self.resolve_kinematic_collision(contact, J)
                    self.apply_friction_impulse(contact, normal, J)

    def apply_friction_impulse(self, contact, normal, Jn):
        """
        Applies Coulomb friction impulse based on the relative velocity,
        including angular friction effect.
        """
        rb1 = contact["rb1"]
        rb2 = contact["rb2"]
        contact_point = contact["contact_point"]  # required for torque calculation

        v1 = rb1.velocity if rb1 and not rb1.isKinematic else Vector3(0, 0, 0)
        v2 = rb2.velocity if rb2 and not rb2.isKinematic else Vector3(0, 0, 0)
        v_rel = v1 - v2

        tangent = v_rel - normal * v_rel.dot(normal)
        tangent_length = tangent.magnitude()

        if tangent_length < 1e-6:
            return  # No significant tangential motion

        tangent = tangent.normalized()

        # Friction coefficient
        if rb1 and rb2:
            mu = rb1._get_friction(rb2)
        elif rb1:
            mu = rb1.friction_coefficient
        elif rb2:
            mu = rb2.friction_coefficient
        else:
            mu = Rigidbody._default_friction

        # Compute friction impulse scalar
        Jt_magnitude = -v_rel.dot(tangent)
        denom = 0.0
        if rb1 and not rb1.isKinematic:
            denom += 1.0 / rb1.mass
        if rb2 and not rb2.isKinematic:
            denom += 1.0 / rb2.mass

        if denom == 0.0:
            return

        Jt_magnitude /= denom
        max_friction = mu * Jn
        Jt_magnitude = max(-max_friction, min(Jt_magnitude, max_friction))

        Jt = tangent * Jt_magnitude

        # Apply linear and angular friction impulses
        if rb1 and not rb1.isKinematic:
            rb1.velocity += Jt / rb1.mass
            r1 = contact_point - rb1.position
            angular_impulse1 = r1.cross(Jt)
            # rb1.angular_velocity += Vector3.from_np(rb1.inverse_inertia @ angular_impulse1.to_np())

        if rb2 and not rb2.isKinematic:
            rb2.velocity -= Jt / rb2.mass
            r2 = contact_point - rb2.position
            angular_impulse2 = r2.cross(-Jt)
            # rb2.angular_velocity += Vector3.from_np(rb2.inverse_inertia @ angular_impulse2.to_np())

    # def resolve_dynamic_collision(self, contact, J):
    #     """
    #     Applies impulse to both dynamic bodies, factoring restitution.
    #     """
    #     n = contact["normal"]
    #     rb1 = contact["rb1"]
    #     rb2 = contact["rb2"]
    #
    #     # Default restitution
    #     restitution = 0.0
    #     if rb1 and rb2:
    #         restitution = min(rb1.restitution, rb2.restitution)
    #     elif rb1:
    #         restitution = rb1.restitution
    #     elif rb2:
    #         restitution = rb2.restitution
    #
    #     # Adjust impulse by restitution
    #     J *= (1 + restitution)
    #
    #     impulse_vec = n * J
    #
    #     if rb1 and not rb1.isKinematic:
    #         rb1.velocity += impulse_vec / rb1.mass
    #     if rb2 and not rb2.isKinematic:
    #         rb2.velocity -= impulse_vec / rb2.mass

    def resolve_dynamic_collision(self, contact, J):
        """
        Applies linear and angular impulse to both dynamic bodies, factoring restitution.
        """
        n = contact["normal"]
        contact_point = contact["point"]  # world-space contact point
        rb1 = contact["rb1"]
        rb2 = contact["rb2"]

        restitution = 0.0
        if rb1 and rb2:
            restitution = min(rb1.restitution, rb2.restitution)
        elif rb1:
            restitution = rb1.restitution
        elif rb2:
            restitution = rb2.restitution

        J *= (1 + restitution)
        impulse_vec = n * J
        # impulse_vec = np.maximum(impulse_vec, 0.0)
        if rb1 and not rb1.isKinematic:
            rb1.velocity += impulse_vec / rb1.mass

            # Angular impulse for rb1
            r1 = contact_point - rb1.position  # lever arm
            # angular_impulse1 = r1.cross(impulse_vec)
            # rb1.angular_velocity += Vector3.from_np(rb1.inverse_inertia @ angular_impulse1.to_np())

        if rb2 and not rb2.isKinematic:
            rb2.velocity -= impulse_vec / rb2.mass

            # Angular impulse for rb2
            r2 = contact_point - rb2.position
            # angular_impulse2 = r2.cross(impulse_vec)
            # rb2.angular_velocity += Vector3.from_np(rb2.inverse_inertia @ angular_impulse2.to_np())

    def resolve_kinematic_collision(self, contact, J):
        """
        Applies linear and angular impulse to the dynamic body only, factoring restitution.
        """
        n = contact["normal"]
        contact_point = contact["contact_point"]  # world-space contact point
        rb1 = contact["rb1"]
        rb2 = contact["rb2"]

        restitution = 0.0
        if rb1 and rb2:
            restitution = min(rb1.restitution, rb2.restitution)
        elif rb1:
            restitution = rb1.restitution
        elif rb2:
            restitution = rb2.restitution

        J *= (1 + restitution)
        impulse_vec = n * J


        if rb1 and not rb1.isKinematic:
            rb1.velocity += impulse_vec / rb1.mass

            # Angular impulse for rb1
            r1 = contact_point - rb1.position
            angular_impulse1 = r1.cross(impulse_vec)
            R = rb1.parent.quaternion.to_matrix3()  # Convert quaternion to 3×3 rotation matrix
            I_inv_world = R @ rb1.inverse_inertia @ R.T

            rb1.angular_velocity += Vector3.from_np(I_inv_world @ angular_impulse1.to_np())

        elif rb2 and not rb2.isKinematic:
            rb2.velocity -= impulse_vec / rb2.mass

            # Angular impulse for rb2
            r2 = contact_point - rb2.position
            angular_impulse2 = r2.cross(-impulse_vec)
            rb2.angular_velocity += Vector3.from_np(rb2.inverse_inertia @ angular_impulse2.to_np())

    # def Stage3(self,children):
    #
    #
    #     if self.parent == None:  # the world does not need an update
    #         for child in children:
    #             child.Stage3(children)
    #         return
    #     rb = self.get_component("Rigidbody")
    #
    #     if rb is None or rb.isKinematic:
    #         return
    #     # total_normal_force = Vector3(0,0,0)
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
    #             normal_force = self.Rigidbody.force.reduce_vector_along_direction(normal * -1) * -1
    #             normal_force = normal * self.Rigidbody.force * -1
    #             # total_normal_force += normal_force
    #             if child.get_component("Rigidbody") is None or child.Rigidbody.isKinematic:
    #                 self.resolve_kinematic_collision(child, normal, contact_point, normal_force)
    #             else:
    #                 self.resolve_dynamic_collision(child, normal, contact_point, normal_force)
    #             # === After updating velocity with all forces ===
    #
    #     # if total_normal_force != Vector3(0,0,0):
    #     #     self.apply_friction(total_normal_force, dt)
    #     # print()

    # def Stage3(self, children):
    #
    #     if self.parent == None:  # the world does not need an update
    #         for child in children:
    #             child.Stage3(children)
    #         return
    #     rb = self.get_component("Rigidbody")
    #
    #     if rb is None or rb.isKinematic:
    #         return
    #     normal_forces = Vector3()
    #     for child in children:
    #         if child == self:
    #             continue  # Skip self
    #
    #         if child.get_component("collider") is None:
    #             continue  # No collider to check against
    #
    #         result = self.collider.check_collision(child)
    #         if result is None:
    #             continue  # No collision
    #
    #         contact_point, normal = result
    #
    #         # Project the force onto the collision normal
    #         force_along_normal = normal * rb.force.dot(normal) * -1
    #
    #         normal_forces += force_along_normal
    #         if child.get_component("Rigidbody") is None or child.Rigidbody.isKinematic:
    #             self.resolve_kinematic_collision(child, normal, contact_point, force_along_normal)
    #         else:
    #             self.resolve_dynamic_collision(child, normal, contact_point, force_along_normal)
    #
    #     rb.normal_forces = normal_forces

    def Stage4(self, children):

        if self.parent == None:  # the world does not need an update
            for child in children:
                child.Stage4(children)
            return
        rb = self.get_component("Rigidbody")

        if rb is None or rb.isKinematic:
            return
        rb.force += rb.normal_forces
        if rb.normal_forces != Vector3(0, 0, 0):
            self.apply_friction(rb.normal_forces, dt)
        rb.normal_forces = Vector3()

    def solve_joints(self, children, dt):
        """
        Go through all objects, find joints, and solve their constraints.
        """
        for child in children:
            joint = child.get_component("joint")
            if joint is not None:
                joint.solve(dt)
    def Start(self):
        children1 = self.get_all_children_bereshit()
        for child in children1:
            for component in child.components.values():
                if hasattr(component, 'Start') and component.Start is not None:
                    try:
                        component.Start()
                    except Exception as e:
                        print(f"[Error] Exception in {component.__class__.__name__}.Start(): {e}")
    def update(self, dt,chack =True):
        children1 = self.get_all_children_bereshit()
        if chack :
            for child in children1:
                for component in child.components.values():
                        if hasattr(component, 'Update') and component.Update is not None:
                            try:
                                component.Update()
                            except Exception as e:
                                print(f"[Error] Exception in {component.__class__.__name__}.Update(): {e}")

        children =self.get_all_children_physics()
        self.Stage1(children) # APPLY GRAVITY and external forces
        self.Stage3(children,dt) # handel collisions and friction
        self.solve_joints(children, dt)

        for child in children:
            rb = child.get_component("Rigidbody")
            if rb is not None:
                child.integrat(dt)

        for child in children1:
            child.rotation = child.quaternion.to_euler()

    def apply_friction(self, normal_force, dt):
        rb = self.Rigidbody
        friction_limit = normal_force.magnitude() * rb.friction_coefficient

        if rb.velocity.magnitude() > 0:
            # Body is sliding, apply kinetic friction
            d = rb.velocity.normalized() * -1
            friction_acc = d * (friction_limit / rb.mass)
            new_velocity = rb.velocity + friction_acc * dt
            if new_velocity.dot(rb.velocity) <= 0:
                rb.velocity = Vector3(0, 0, 0)
            else:
                rb.velocity = new_velocity

        elif rb.force.magnitude() > 0:
            # Body is at rest, apply static friction
            d = rb.force.normalized() * -1
            applied_force_magnitude = rb.force.magnitude()
            if applied_force_magnitude <= friction_limit:
                # Static friction cancels all force
                rb.force = Vector3(0, 0, 0)
            else:
                # Static friction resists up to limit, remainder causes movement
                rb.force += d * friction_limit

    def integrat(self, dt):

        # === 4) INTEGRATION PHASE ===
        # 4.1) Linear acceleration & velocity:
        self.Rigidbody.acceleration = self.Rigidbody.force / self.Rigidbody.mass
        # self.Rigidbody.acceleration.y *= self.Rigidbody.drag

        self.Rigidbody.velocity += self.Rigidbody.acceleration * dt
        # 4.2) Angular acceleration & velocity (component‐wise):
        self.Rigidbody.angular_acceleration = Vector3(
            self.Rigidbody.torque.x / self.Rigidbody.inertia.x if self.Rigidbody.inertia.x != 0 else 0,
            self.Rigidbody.torque.y / self.Rigidbody.inertia.y if self.Rigidbody.inertia.y != 0 else 0,
            self.Rigidbody.torque.z / self.Rigidbody.inertia.z if self.Rigidbody.inertia.z != 0 else 0
        )
        self.Rigidbody.angular_velocity += self.Rigidbody.angular_acceleration * dt
        # self.Rigidbody.angular_velocity *= self.Rigidbody.drag
        # 4.3) Integrate rotation
        ang_disp = self.Rigidbody.angular_velocity * dt \
                   + 0.5 * self.Rigidbody.angular_acceleration * dt * dt

        # self.quaternion *=
        self.quaternion *= Quaternion.euler(ang_disp)


        self.position += self.Rigidbody.velocity * dt \
                         + 0.5 * self.Rigidbody.acceleration * dt * dt

        self.Rigidbody.force = Vector3(0, 0, 0)
        self.Rigidbody.torque = Vector3(0, 0, 0)
        # self.add_rotation(ang_disp)
        children = self.get_all_children_not_physics()
        for child in children:
            child.position += self.Rigidbody.velocity * dt \
                         + 0.5 * self.Rigidbody.acceleration * dt * dt

        # if self.get_component("joint") != None:
        #     if self.joint.look_position:
        #         self.joint.other.position += self.Rigidbody.velocity * dt \
        #                          + 0.5 * self.Rigidbody.acceleration * dt * dt
        # 4.5) Angular damping
        # self.Rigidbody.angular_velocity *= 0.98

        # 4.6) Reset torques for next frame

        self.Rigidbody.angular_acceleration = Vector3(0, 0, 0)
        self.Rigidbody.torque = Vector3(0, 0, 0)
    # def integrat2(self, dt):
    #
    #     # === 4) INTEGRATION PHASE ===
    #     self.Rigidbody.velocity += self.Rigidbody.acceleration * dt
    #     # 4.2) Angular acceleration & velocity (component‐wise):
    #     self.Rigidbody.angular_acceleration = Vector3(
    #         self.Rigidbody.torque.x / self.Rigidbody.inertia.x if self.Rigidbody.inertia.x != 0 else 0,
    #         self.Rigidbody.torque.y / self.Rigidbody.inertia.y if self.Rigidbody.inertia.y != 0 else 0,
    #         self.Rigidbody.torque.z / self.Rigidbody.inertia.z if self.Rigidbody.inertia.z != 0 else 0
    #     )
    #     self.Rigidbody.angular_velocity += self.Rigidbody.angular_acceleration * dt
    #
    #     # 4.3) Integrate rotation
    #     ang_disp = self.Rigidbody.angular_velocity * dt \
    #                + 0.5 * self.Rigidbody.angular_acceleration * dt * dt
    #     if self.get_component("joint") != None:
    #         if not self.joint.look_position:
    #             # 4.4) Integrate position
    #
    #             self.position += self.Rigidbody.velocity * dt \
    #                              + 0.5 * self.Rigidbody.acceleration * dt * dt
    #             self.set_position(self.position)
    #         if not self.joint.look_rotation:
    #             self.add_rotation(ang_disp)
    #     else:
    #         # 4.4) Integrate position
    #         self.position += self.Rigidbody.velocity * dt \
    #                          + 0.5 * self.Rigidbody.acceleration * dt * dt
    #
    #         self.set_position(self.position)
    #
    #         self.add_rotation(ang_disp)
    #
    #     # 4.5) Angular damping
    #     # self.Rigidbody.angular_velocity *= 0.98
    #
    #     # 4.6) Reset torques for next frame
    #     self.Rigidbody.angular_acceleration = Vector3(0, 0, 0)
    #     self.Rigidbody.torque = Vector3(0, 0, 0)

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

    def set_rotation(self, new_world_rot: Vector3):
        """
        Explicitly set this object's world‐space rotation to `new_world_rot`.
        Also recompute `local_rotation` so that:
            local_rotation = new_world_rot - parent.rotation
        (or = new_world_rot if there's no parent).
        """
        # 1. Assign the new world rotation
        self.rotation = Vector3(new_world_rot.x, new_world_rot.y, new_world_rot.z)
        self._rotation_dirty = True
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

    def add_rotation(self, delta: Vector3,forall=False):
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
            rotated_local = rotate_vector_old(self.local_position,Vector3(0,0,0), self.parent.rotation)
            self.position = self.parent.position + rotated_local

        # Recursively update children
        for child in self.children:
            child.update_world_transform()

    def add_rotation_old(self, rotation):
        self.set_rotation(self.rotation + rotation)

    # def set_rotation_old(self, rotation):
    #     delta = rotation - self.rotation
    #     if delta != Vector3(0, 0, 0):
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
            # target.set_position(Vector3(*rotated))
            target.position = Vector3(*rotated)
            target.set_projection(pivot=pivot)

    def find_center_of_gravity(self):
        bereshit = self.get_all_children_bereshit()
        positions = [obj.position.to_tuple() for obj in bereshit]
        masses = [obj.Rigidbody.mass for obj in bereshit]
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
        if self.get_component("Rigidbody") is not None:
            self.Rigidbody.acceleration = Vector3(0,0,0)
            self.Rigidbody.velocity = Vector3(0,0,0)

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
    def get_all_children_physics(self):
        all_objs = []
        for child in self.children:
            rb = child.get_component("Rigidbody")
            collider = child.get_component("collider")
            if rb and collider:
                all_objs.append(child)
            all_objs.extend(child.get_all_children_physics())
        return all_objs
    def get_all_children_not_physics(self):
        all_objs = []
        for child in self.children:
            rb = child.get_component("Rigidbody")
            # collider = child.get_component("collider")
            if not rb:
                all_objs.append(child)
            all_objs.extend(child.get_all_children_physics())
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

def compute_axes_from_euler(rotation):
    """
    Given Euler angles (rotation.x, rotation.y, rotation.z) in degrees,
    compute the rotated basis axes: right, up, forward.

    Returns:
        [right: Vector3, up: Vector3, forward: Vector3]
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
    right = Vector3(*R[:,0])
    up = Vector3(*R[:,1])
    forward = Vector3(*R[:,2])

    return [right, up, forward]
def get_rotation_matrix(angles):
    angles = np.array([angles.x, angles.y, angles.z])
    angle_x, angle_y, angle_z = np.radians(-angles)

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    R_y = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    R_z = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])

    # Same convention: X, then Y, then Z
    R = R_z @ R_y @ R_x
    return R

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

    return Vector3(temp[0], temp[1], temp[2])
def rotate_vector(vector, pivot, rotation: Quaternion):
    """
    Rotate 'vector' around 'pivot' using quaternion 'rotation' (Vector4).
    """
    # Convert to numpy for convenience
    v_np = np.array([vector.x, vector.y, vector.z])
    p_np = np.array([pivot.x, pivot.y, pivot.z])

    # Compute relative position
    rel_v = v_np - p_np

    # Represent as pure quaternion
    v_quat = Quaternion(0.0, rel_v[0], rel_v[1], rel_v[2])

    # q * v * q^-1
    q_conj = rotation.conjugate()
    rotated_quat = rotation.multiply(v_quat).multiply(q_conj)

    # Extract rotated vector part
    rotated_v = np.array([rotated_quat.x, rotated_quat.y, rotated_quat.z])

    # Shift back
    result = rotated_v + p_np

    return Vector3(result[0], result[1], result[2])


def rotate_vector_quaternion(vector, quaternion):
    q = quaternion.normalized()
    q_conj = q.conjugate()

    # Pure quaternion from vector: x, y, z go into x, y, z, and w = 0
    vec_quat = Quaternion(vector.x, vector.y, vector.z, 0)

    # Apply rotation: q * v * q_conj
    rotated = q * vec_quat * q_conj

    # Extract rotated vector part
    return Vector3(rotated.x, rotated.y, rotated.z)


# def inverse_rotate_vector(vector, quaternion_angles):
#     inverse_quaternion = Vector3(-quaternion_angles.x, -quaternion_angles.y, -quaternion_angles.z)
#     return rotate_vector(vector, Vector3(0, 0, 0), inverse_quaternion)

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
