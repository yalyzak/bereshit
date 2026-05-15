import numpy as np

from bereshit.Physics import RaycastHit
from bereshit.Vector3 import Vector3
from bereshit.Quaternion import Quaternion

class Collider:
    Scale = 1

    def __init__(self, size=None, position=None, rotation=Vector3(), object_pointer=None, is_trigger=False):
        self.__delta_size = Vector3() if not size else size
        self.__delta_position = Vector3() if not position else position
        self.__delta_quaternion = Quaternion.euler(rotation)

        self.obj = object_pointer
        self.is_trigger = is_trigger
        self.enter = False
        self.stay = False

    @property
    def size(self):
        return self.__delta_size + self.parent.size

    @property
    def position(self):
        return self.__delta_position + self.parent.position

    @property
    def quaternion(self):
        return self.__delta_quaternion * self.parent.quaternion

    @staticmethod
    def check_collision(collider1, collider2, single_point=False, collided_a=True, collided_b=True):
        pass

    @staticmethod
    def aabb_collision(obj1, obj2):
        min1, max1 = obj1.get_aabb()
        min2, max2 = obj2.get_aabb()

        # Check overlap on all 3 axes
        return (
                (min1.x <= max2.x and max1.x >= min2.x) and
                (min1.y <= max2.y and max1.y >= min2.y) and
                (min1.z <= max2.z and max1.z >= min2.z)
        )

    def get_aabb(self):
        # Half extents
        half = self.size * 0.5

        abs_rot = self.quaternion.to_matrix3_abs(self.parent.Cache.R)

        # Compute world extents
        world_half = half.MatrixMultiplication(abs_rot)

        # AABB min/max
        min_corner = self.position - world_half
        max_corner = self.position + world_half

        return min_corner, max_corner

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

    @staticmethod
    def handle_collision_exit(self, other_collider, collided_a, collided_b):
        collision1 = Collision(self, None)
        collision2 = Collision(other_collider, None)
        if (self.stay or self.enter) and not collided_a:
            self.OnCollisionExit(collision2)
        if (other_collider.stay or other_collider.enter) and not collided_b:
            other_collider.OnCollisionExit(collision1)

    @staticmethod
    def handle_collision_events(self, other_collider, result):
        collision1 = Collision(self, result)
        collision2 = Collision(other_collider, result)

        if self.is_trigger:
            self.OnTriggerEnter(collision2)
        if other_collider.is_trigger:
            other_collider.OnTriggerEnter(collision1)
        if self.enter == False:
            self.OnCollisionEnter(collision2)
        else:
            self.OnCollisionStay(collision2)
        if other_collider.enter == False:
            other_collider.OnCollisionEnter(collision1)
        else:
            other_collider.OnCollisionStay(collision1)

    def Raycast(self, origin, direction, maxDistance=float('inf'), hit=None):
        print(f"Ray casting was not defined for {self.__class__.__name__}")
        return RaycastHit()


class ContactPoints:
    def __init__(self, contact_points, normal, depth):
        self.contact_points = contact_points
        self.normal = normal
        self.depth = depth

    @staticmethod
    def average_point(contact_points):
        count = len(contact_points)
        if count:
            total_p = Vector3(0, 0, 0)

            for point in contact_points:
                total_p += point

            avg_p = total_p / count

            return avg_p
        return None


class Collision:
    def __init__(self, other, contactPoints):
        self.other = other
        self.contactPoints = contactPoints
