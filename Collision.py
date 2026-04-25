from bereshit.Vector3 import Vector3


class Collision:
    def __init__(self, other, normal):
        self.normal = normal
        self.other = other
