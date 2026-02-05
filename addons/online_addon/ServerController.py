import time
import keyboard
from bereshit import Vector3


class ServerController:
    def __init__(self, speed=5, speed2=15):
        self.force_amount = speed
        self.force_amount2 = speed2


    def server_controller(self, dt, key):

        if key == 'w':
            forward = self.parent.quaternion.rotate(Vector3(0, 0, 1))
            # flatten to XZ plane
            forward = Vector3(forward.x, 0, forward.z).normalized() * self.force_amount
            self.parent.Rigidbody.velocity += forward * dt

        if key == 's':
            backward = self.parent.quaternion.rotate(Vector3(0, 0, -1))
            backward = Vector3(backward.x, 0, backward.z).normalized() * self.force_amount
            self.parent.Rigidbody.velocity += backward * dt

        if key == 'a':
            right = self.parent.quaternion.rotate(Vector3(1, 0, 0))
            right = Vector3(right.x, 0, right.z).normalized() * self.force_amount
            self.parent.Rigidbody.velocity += right * dt

        if key == 'd':
            left = self.parent.quaternion.rotate(Vector3(-1, 0, 0))
            left = Vector3(left.x, 0, left.z).normalized() * self.force_amount
            self.parent.Rigidbody.velocity += left * dt

        if keyboard.is_pressed('space'):
            self.parent.Rigidbody.velocity += Vector3(0, self.force_amount2, 0) * dt

        if keyboard.is_pressed('left shift'):
            self.parent.Rigidbody.velocity += Vector3(0, -self.force_amount2, 0) * dt

