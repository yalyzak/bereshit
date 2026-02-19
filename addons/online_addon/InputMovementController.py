from bereshit import Vector3


class InputMovementController:
    def __init__(self, speed=5, speed2=15):
        self.force_amount = speed
        self.force_amount2 = speed2

    def move_with_keys(self, dt, keys):
        """
        keys: iterable (list/set) of pressed key strings
        example: ['w', 'a', 'space']
        """

        # Forward
        if 'w' in keys:
            forward = self.parent.quaternion.rotate(Vector3(0, 0, 1))
            forward = Vector3(forward.x, 0, forward.z).normalized() * self.force_amount
            self.parent.Rigidbody.velocity += forward * dt

        # Backward
        if 's' in keys:
            backward = self.parent.quaternion.rotate(Vector3(0, 0, -1))
            backward = Vector3(backward.x, 0, backward.z).normalized() * self.force_amount
            self.parent.Rigidbody.velocity += backward * dt

        # Right
        if 'a' in keys:
            right = self.parent.quaternion.rotate(Vector3(1, 0, 0))
            right = Vector3(right.x, 0, right.z).normalized() * self.force_amount
            self.parent.Rigidbody.velocity += right * dt

        # Left
        if 'd' in keys:
            left = self.parent.quaternion.rotate(Vector3(-1, 0, 0))
            left = Vector3(left.x, 0, left.z).normalized() * self.force_amount
            self.parent.Rigidbody.velocity += left * dt

        # Up
        if 'space' in keys:
            self.parent.Rigidbody.velocity += Vector3(0, self.force_amount2, 0) * dt

        # Down
        if 'left shift' in keys:
            self.parent.Rigidbody.velocity += Vector3(0, -self.force_amount2, 0) * dt

    def Update(self, dt, keys):
        self.move_with_keys(dt, keys)
