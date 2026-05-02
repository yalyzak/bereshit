from bereshit.Vector3 import Vector3





class CollisionRigidbody:
    def __init__(self, rb1, rb2, normal, contact_point):
        self.rb1 = rb1
        self.rb2 = rb2
        self.normal = normal
        self.contact_point = contact_point
        self.r1 = contact_point - rb1.parent.position
        self.r2 = contact_point - rb2.parent.position


class ContactPoint:
    def __init__(self, point, normal, depth):
        self.point = point
        self.normal = normal
        self.depth = depth
