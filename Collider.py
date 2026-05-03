from bereshit.Physics import RaycastHit


class Collider:
    Scale = 1

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

    @staticmethod
    def exit_flags(self, other_collider, collided_a, collided_b):
        collision1 = Collision(self, None)
        collision2 = Collision(other_collider, None)
        if (self.stay or self.enter) and not collided_a:
            self.OnCollisionExit(collision2)
        if (other_collider.stay or other_collider.enter) and not collided_b:
            other_collider.OnCollisionExit(collision1)

    @staticmethod
    def collision_flags(self, other_collider, result):
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
