class Vector2:
    __slots__ = ("x", "y")

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    @classmethod
    def from_np(cls, arr):
        """Create a Vector3 from a NumPy array or list."""
        return cls(arr[0], arr[1])