import time

import numpy as np

from bereshit.Vector3 import Vector3
from bereshit.Quaternion import Quaternion
from bereshit.Physics import RaycastHit
from bereshit.class_type import Collider
from bereshit.Collision import Collision


class BoxCollider(Collider):
    def __init__(self, size=None, rotation=None, object_pointer=None, is_trigger=False):
        self.size = size
        self.rotation = rotation

        self.obj = object_pointer
        self.is_trigger = is_trigger
        self.enter = False
        self.stay = False

    def OnCollisionEnter(self, collision):
        self.enter = True

        for component in self.parent.components.values():
            if hasattr(component, 'OnCollisionEnter') and component.OnCollisionEnter is not None and component != self:
                component.OnCollisionEnter(collision)

    def OnCollisionStay(self, collision):
        self.enter = False
        self.stay = True
        for component in self.parent.components.values():
            if hasattr(component, 'OnCollisionStay') and component.OnCollisionStay is not None and component != self:
                component.OnCollisionStay(collision)

    def OnCollisionExit(self, collision):
        self.enter = False
        self.stay = False

        for component in self.parent.components.values():
            if hasattr(component, 'OnCollisionExit') and component.OnCollisionExit is not None and component != self:
                component.OnCollisionExit(collision)

    def OnTriggerEnter(self, collision):
        """This method can be overwritten by subclasses to handle trigger events."""
        for component in self.parent.components.values():
            if hasattr(component, 'OnTriggerEnter') and component.OnTriggerEnter is not None and component != self:
                component.OnTriggerEnter(collision)

    def _find_contaact_points_raycast(self, other_collider, collision_axis):
        hits = set()
        edges = other_collider.edges()
        ver = other_collider.vertices()

        for edge in edges:
            if len(hits) < 4:
                start, end = edge
                vector = ver[end] - ver[start]
                norm = np.linalg.norm(vector)
                hit = self.Raycast(ver[start], vector / norm, maxDistance=norm)
                if hit.point is not None:
                    V = Vector3(tuple(
                        hit.point)) - self.parent.position
                    if V.dot(collision_axis) > 0:
                        collision_axis = -collision_axis
                    hits.add((tuple(hit.point), -collision_axis, 0))
                rb = (self, other_collider)

        if len(hits) == 0:
            ver = self.vertices()
            for edge in edges:
                # if len(hits) == 4:
                #     continue
                start, end = edge
                vector = ver[end] - ver[start]
                norm = np.linalg.norm(vector)
                hit = other_collider.Raycast(ver[start], vector / norm, maxDistance=norm)
                if hit.point is not None:
                    V = Vector3(tuple(
                        hit.point)) - other_collider.parent.position  # i think that maybe i need to add if collision_type == b then use self.other_collider.position
                    if V.dot(collision_axis) > 0:
                        collision_axis = -collision_axis
                    hits.add((tuple(hit.point), -collision_axis, 0))
                    rb = (other_collider, self)
        contact_points = list(hits)
        return contact_points, rb

    def _find_contaact_points(self, a_axes, a_center, a_half, b_axes, b_center, b_half, collision_type,
                              collision_axis_indices, other):

        if collision_type == "a" or collision_type == "b":
            return self._FaceToFace(a_axes, a_center, a_half, b_axes, b_center, b_half, collision_type,
                                    collision_axis_indices, other)

        else:
            [None, (other, self)]

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
        world_corners = [rotate_vector_old(corner, self.obj.position, self.obj.rotation) + self.obj.position for corner
                         in
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

    def vertices(self):
        center, size, rotation = self.parent.position, self.size, self.obj.quaternion.conjugate()
        R = rotation.to_matrix3()
        w, h, d = np.array(size) / 2
        corners = np.array([
            [-w, -h, -d],
            [w, -h, -d],
            [w, h, -d],
            [-w, h, -d],
            [-w, -h, d],
            [w, -h, d],
            [w, h, d],
            [-w, h, d],
        ])
        rotated = corners @ R.T
        return rotated + center.to_np()

    def triangles(self):
        vertices = self.vertices()
        # --- Define triangles using vertex indices ---
        # Each triplet = one triangle
        tris_idx = [
            # Front face (-Z)
            [0, 1, 2], [0, 2, 3],
            # Back face (+Z)
            [4, 6, 5], [4, 7, 6],
            # Left face (-X)
            [0, 3, 7], [0, 7, 4],
            # Right face (+X)
            [1, 5, 6], [1, 6, 2],
            # Bottom face (-Y)
            [0, 4, 5], [0, 5, 1],
            # Top face (+Y)
            [3, 2, 6], [3, 6, 7],
        ]

        # --- Build triangle vertex arrays ---
        triangles = [vertices[i] for i in tris_idx]
        triangles = [np.array([vertices[a], vertices[b], vertices[c]]) for a, b, c in tris_idx]

        return triangles

    def edges(self):
        return np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ])

    def faces(self):

        """
        Returns the 6 faces of the box as arrays of 4 vertices each.
        The vertex order for each face is counter-clockwise when looking at the face.
        """
        v = self.vertices()

        # Define faces using vertex indices (each list = one face of 4 corners)
        face_indices = [
            # Front (-Z)
            [0, 1, 2, 3],
            # Back (+Z)
            [4, 5, 6, 7],
            # Left (-X)
            [0, 3, 7, 4],
            # Right (+X)
            [1, 5, 6, 2],
            # Bottom (-Y)
            [0, 4, 5, 1],
            # Top (+Y)
            [3, 2, 6, 7],
        ]

        # Build the faces as arrays of vertex positions
        faces = [np.array([v[i] for i in face]) for face in face_indices]
        return faces

    def temp(self, center, half_size, R):
        faces = []
        axes = [R[:, 0], R[:, 1], R[:, 2]]  # local x, y, z directions
        names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

        for i, axis in enumerate(axes):
            for sign in [1, -1]:
                # Face center: move from box center along the face normal
                face_center = center + sign * axis * half_size[i]

                # Half-size of face: drop the normal axis
                face_half = np.delete(half_size, i)

                # Build rotation matrix for the face (tangential axes)
                tangential_axes = [axes[j] for j in range(3) if j != i]
                face_R = np.column_stack(tangential_axes + [sign * axis])  # x,y,toward normal

                faces.append({
                    "name": f"{'+' if sign > 0 else '-'}{['X', 'Y', 'Z'][i]}",
                    "center": face_center,
                    "half_size": face_half,
                    "rotation": face_R
                })
        return faces

    def Raycast(self, origin, direction, maxDistance=float('inf'), hit=None):
        def ray_box_intersect(orig, dir, box_min, box_max, eps=1e-8):
            inv_dir = 1.0 / (dir + eps * (dir == 0.0))

            tx1 = (box_min[0] - orig[0]) * inv_dir[0]
            tx2 = (box_max[0] - orig[0]) * inv_dir[0]

            tmin = min(tx1, tx2)
            tmax = max(tx1, tx2)
            hit_axis = 0

            ty1 = (box_min[1] - orig[1]) * inv_dir[1]
            ty2 = (box_max[1] - orig[1]) * inv_dir[1]

            tymin = min(ty1, ty2)
            tymax = max(ty1, ty2)

            if tymin > tmin:
                tmin = tymin
                hit_axis = 1
            tmax = min(tmax, tymax)

            tz1 = (box_min[2] - orig[2]) * inv_dir[2]
            tz2 = (box_max[2] - orig[2]) * inv_dir[2]

            tzmin = min(tz1, tz2)
            tzmax = max(tz1, tz2)

            if tzmin > tmin:
                tmin = tzmin
                hit_axis = 2
            tmax = min(tmax, tzmax)

            if tmax < 0 or tmin > tmax:
                return None, None

            hit_point = orig + dir * tmin

            normal = [0.0, 0.0, 0.0]
            normal[hit_axis] = -1.0 if dir[hit_axis] > 0 else 1.0

            return hit_point, np.array(normal)

        def ray_obb_intersect(orig, dir, center, half_size, rotation_matrix):
            """
            Ray-OBB intersection.
            Returns: (hit_world, normal_world) or (None, None)
            """
            # Transform ray into box local space
            local_orig = rotation_matrix.T @ (orig - center)
            local_dir = rotation_matrix.T @ dir

            # Box extents in local space
            box_min = -half_size
            box_max = half_size

            hit_local, normal_local = ray_box_intersect(local_orig, local_dir, box_min, box_max)

            if hit_local is None:
                return None

            # Transform intersection and normal back to world space
            hit_world = (rotation_matrix @ hit_local) + center
            normal_world = rotation_matrix @ normal_local
            normal_world /= np.linalg.norm(normal_world)

            return RaycastHit(hit_world, normal_world, np.linalg.norm(hit_world - orig), self)

        center = self.parent.position.to_np()
        half_size = (self.parent.size / 2).to_np()
        R = self.parent.quaternion.conjugate().to_matrix3()

        faces = self.temp(center, half_size, R)

        dis = float('inf')
        hit = RaycastHit()
        for face in faces:

            temp_hit = ray_obb_intersect(origin, direction,
                                         face["center"],
                                         np.array([*face["half_size"], 0]),  # make it 3D if needed
                                         face["rotation"])

            if temp_hit is not None:
                if temp_hit.distance < dis and temp_hit.distance < maxDistance:
                    hit.distance = temp_hit.distance
                    hit.point = temp_hit.point
                    hit.normal = temp_hit.normal
                    hit.collider = temp_hit.collider

        return hit

    # @staticmethod
    def _FaceToFace(self, a_axes, a_center, a_half, b_axes, b_center, b_half, collision_type, collision_axis_indices,
                    other):
        if collision_type == "a":
            ref_axes, ref_center, ref_half = a_axes, a_center, a_half
            inc_axes, inc_center, inc_half = b_axes, b_center, b_half
            flip = False
        else:
            ref_axes, ref_center, ref_half = b_axes, b_center, b_half
            inc_axes, inc_center, inc_half = a_axes, a_center, a_half
            flip = True

        def half_on_axis(h, i):
            return h.x if i == 0 else h.y if i == 1 else h.z

        face_index = collision_axis_indices
        ref_normal = ref_axes[face_index]
        if flip:
            ref_normal = -ref_normal

        # --- find incident face ---
        incident_face = max(
            range(3),
            key=lambda i: abs(inc_axes[i].dot(ref_normal))
        )

        # --- reference plane ---
        plane_point = ref_center + ref_normal * half_on_axis(ref_half, face_index)

        # --- incident face vertices ---
        def get_face_vertices(center, axes, half, i):
            j = (i + 1) % 3
            k = (i + 2) % 3

            sign = -1 if axes[i].dot(ref_normal) > 0 else 1
            verts = []

            for sj in (-1, 1):
                for sk in (-1, 1):
                    verts.append(
                        center
                        + axes[i] * sign * half_on_axis(half, i)
                        + axes[j] * sj * half_on_axis(half, j)
                        + axes[k] * sk * half_on_axis(half, k)
                    )
            return verts

        contacts = get_face_vertices(inc_center, inc_axes, inc_half, incident_face)

        # --- clip ---
        def clip_polygon(poly, n, p):
            out = []
            for i in range(len(poly)):
                A = poly[i]
                B = poly[(i + 1) % len(poly)]
                da = (A - p).dot(n)
                db = (B - p).dot(n)

                if da <= 0:
                    out.append(A)
                if da * db < 0:
                    t = da / (da - db)
                    out.append(A + (B - A) * t)
            return out

        i = face_index
        for axis_idx in ((i + 1) % 3, (i + 2) % 3):
            axis = ref_axes[axis_idx]
            limit = half_on_axis(ref_half, axis_idx)
            for sign in (-1, 1):
                contacts = clip_polygon(
                    contacts,
                    axis * sign,
                    ref_center + axis * sign * limit
                )

        # --- final contacts ---
        final_contacts = []
        for p in contacts:
            depth = (plane_point - p).dot(ref_normal)
            if depth >= 0:
                final_contacts.append((p, -ref_normal, depth))
                V = p - a_center
                if V.dot(ref_normal) > 0:
                    rb = (other, self)
                else:
                    rb = (self, other)

        return final_contacts, rb

    @staticmethod
    def _get_axes(rotation: Quaternion):
        R = rotation.to_matrix3()
        return [
            Vector3(*R[:, 0]).normalized(),
            Vector3(*R[:, 1]).normalized(),
            Vector3(*R[:, 2]).normalized(),
        ]

    @staticmethod
    def _project_box(center, axes, half_sizes, axis):
        c = center.dot(axis)
        r = sum(abs(axis.dot(a)) * h for a, h in zip(axes, half_sizes))
        return c - r, c + r

    @staticmethod
    def _overlap_on_axis(p1, p2):
        return not (p1[1] < p2[0] or p2[1] < p1[0])

    @staticmethod
    def _average_contact_data(contact_points):
        if not contact_points:
            return None  # or raise Exception

        total_p = Vector3(0, 0, 0)
        total_n = Vector3(0, 0, 0)
        total_d = 0.0

        for p, n, d in contact_points:
            total_p += p
            total_n += n
            total_d += d

        count = len(contact_points)
        avg_p = total_p / count
        avg_n = total_n.normalized()  # normalize the summed normal vector
        avg_d = total_d / count

        return avg_p, avg_n, avg_d

    def _SAT(self, other_collider):
        a_center = self.obj.position
        b_center = other_collider.obj.position

        a_axes = self._get_axes(self.obj.quaternion)
        b_axes = self._get_axes(other_collider.obj.quaternion.conjugate())

        a_half = self.obj.size * 0.5
        b_half = other_collider.obj.size * 0.5

        a_half_sizes = [a_half.x, a_half.y, a_half.z]
        b_half_sizes = [b_half.x, b_half.y, b_half.z]

        axes_to_test = []

        for i in range(3):
            axes_to_test.append(("a", i, a_axes[i]))
            axes_to_test.append(("b", i, b_axes[i]))

        for i in range(3):
            for j in range(3):
                cross = a_axes[i].cross(b_axes[j])
                if cross.magnitude() > 1e-6:
                    axes_to_test.append(("edge", (i, j), cross.normalized()))

        smallest_overlap = float("inf")
        collision_axis = None
        collision_type = None
        collision_axis_indices = None

        for source, indices, axis in axes_to_test:
            proj_a = self._project_box(a_center, a_axes, a_half_sizes, axis)
            proj_b = self._project_box(b_center, b_axes, b_half_sizes, axis)

            if not self._overlap_on_axis(proj_a, proj_b):
                return None

            overlap = min(proj_a[1], proj_b[1]) - max(proj_a[0], proj_b[0])
            if overlap < smallest_overlap:
                smallest_overlap = overlap
                collision_axis = axis
                collision_type = source
                collision_axis_indices = indices

        if (b_center - a_center).dot(collision_axis) < 0:
            collision_axis = -collision_axis

        return a_axes, a_center, a_half, b_axes, b_center, b_half, collision_type, collision_axis_indices, collision_axis

    def check_collision(self, other, single_point=False, collided_a=True, collided_b=True):

        other_collider = getattr(other, 'collider', other)
        if other_collider is None:
            return None
        collision = Collision(other, None)

        result = self._SAT(other_collider)
        if result is None:
            if (self.stay or self.enter) and not collided_a:
                self.OnCollisionExit(collision)
            if (other_collider.stay or other_collider.enter) and not collided_b:
                other_collider.OnCollisionExit(collision)  # could creat problems if two objects collide at once
            return None
        a_axes, a_center, a_half, b_axes, b_center, b_half, collision_type, collision_axis_indices, collision_axis = result

        # result = self._find_contaact_points(a_axes, a_center, a_half, b_axes, b_center, b_half, collision_type, collision_axis_indices, other_collider)
        # if result is None:
        #     return None
        # result, rb = result
        result, rb = self._find_contaact_points_raycast(other_collider, collision_axis)
        collision = Collision(other, -collision_axis)

        if single_point:
            result = self._average_contact_data(result)
        if result is None or result == []:
            return None

        # collision_axis, smallest_overlap, collision_type, collision_axis_indices = result

        if self.is_trigger:
            self.OnTriggerEnter(collision)
        if other_collider.is_trigger:
            other_collider.OnTriggerEnter(collision)

        if self.enter == False:
            self.OnCollisionEnter(collision)
        else:
            self.OnCollisionStay(collision)

        if other_collider.enter == False:
            other_collider.OnCollisionEnter(collision)
        else:
            other_collider.OnCollisionStay(collision)

        # return contact_points, rb
        return result, rb

    def attach(self, owner_object):
        box = owner_object.get_component(BoxCollider)  # will remove duplicate renders by default
        if box:
            owner_object.remove_component("Collider")

        if self.size == None:
            self.size = owner_object.size

        if self.rotation == None:
            self.rotation = owner_object.rotation

        self.obj = owner_object
        return "Collider"  # need to be change to "Collider" but fucks up the whole update loop
