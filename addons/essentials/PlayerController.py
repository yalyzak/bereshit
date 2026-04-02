from bereshit import Vector3

from collections import deque
import keyboard


class PlayerController:
    def __init__(self, speed=5, speed2=100, record_inputs=False, max_queue_size=1000):
        self.force_amount = speed
        self.force_amount2 = speed2



        # Recording settings
        self.record_inputs = record_inputs
        self.input_queue = deque(maxlen=max_queue_size)

    def record_input(self, dt, pressed_keys):
        """
        Store input snapshot for this frame.
        """
        if self.record_inputs:
            self.input_queue.append({
                "dt": dt,
                "keys": pressed_keys.copy()
            })

    def get_next_input(self):
        """
        Remove and return the oldest input in the queue.
        Returns None if empty.
        """
        if self.input_queue:
            return self.input_queue.popleft()
        return None

    def peek_inputs(self):
        """
        Return a copy of all inputs without removing them.
        """
        return list(self.input_queue)

    def clear_inputs(self):
        """
        Clears the queue.
        """
        self.input_queue.clear()

    def keyboard_controller(self, dt):
        pressed_keys = []


        if keyboard.is_pressed('w'):
            pressed_keys.append('w')
            forward = self.parent.quaternion.rotate(Vector3(0, 0, 1))
            forward = Vector3(forward.x, 0, forward.z).normalized() * self.force_amount
            self.parent.Rigidbody.velocity += forward * dt

        if keyboard.is_pressed('s'):
            pressed_keys.append('s')
            backward = self.parent.quaternion.rotate(Vector3(0, 0, -1))
            backward = Vector3(backward.x, 0, backward.z).normalized() * self.force_amount
            self.parent.Rigidbody.velocity += backward * dt

        if keyboard.is_pressed('a'):
            pressed_keys.append('a')
            right = self.parent.quaternion.rotate(Vector3(1, 0, 0))
            right = Vector3(right.x, 0, right.z).normalized() * self.force_amount
            self.parent.Rigidbody.velocity += right * dt

        if keyboard.is_pressed('d'):
            pressed_keys.append('d')
            left = self.parent.quaternion.rotate(Vector3(-1, 0, 0))
            left = Vector3(left.x, 0, left.z).normalized() * self.force_amount
            self.parent.Rigidbody.velocity += left * dt

        if keyboard.is_pressed('space') and (self.parent.Collider.stay or self.parent.Collider.enter):
            pressed_keys.append('space')
            self.parent.Rigidbody.velocity += Vector3(0, self.force_amount2, 0) * dt

        if keyboard.is_pressed('left shift'):
            pressed_keys.append('left shift')
            self.parent.Rigidbody.velocity += Vector3(0, -self.force_amount2, 0) * dt

        # Record input snapshot for this frame
        self.record_input(dt, pressed_keys)

    def Update(self, dt):
        self.keyboard_controller(dt)


