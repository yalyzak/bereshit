class ContactPoint:
    def __init__(self, position, normal, penetration):
        self.position = position
        self.normal = normal
        self.penetration = penetration

        # solver state (cached)
        self.normal_impulse = 0.0
        self.tangent_impulse = 0.0

class ContactManifold:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.contacts = []   # up to 4 for box-box
        self.normal = None
