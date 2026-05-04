import numpy as np

from bereshit.Physics import RaycastHit
from bereshit.Vector3 import Vector3


class Collider:
    Scale = 1

    def __init__(self, size=None, rotation=None, object_pointer=None, is_trigger=False):
        self.size = size
        self.rotation = rotation

        self.obj = object_pointer
        self.is_trigger = is_trigger
        self.enter = False
        self.stay = False

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
        half = self.size.to_np() * 0.5

        rot = self.parent.quaternion.to_matrix3()

        # Take absolute value of rotation matrix
        abs_rot = np.abs(rot)

        # Compute world extents
        world_half = Vector3.from_np(abs_rot @ half)

        # AABB min/max
        min_corner = self.parent.position - world_half
        max_corner = self.parent.position + world_half

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


class Collision:
    def __init__(self, other, contactPoints):
        self.other = other
        self.contactPoints = contactPoints
