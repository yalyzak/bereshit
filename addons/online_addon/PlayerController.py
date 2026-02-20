import math

from bereshit import Vector3, Quaternion

from collections import deque
import keyboard
import mouse

CENTER_X = 960
CENTER_Y = 540
sensitivity = 0.1  # adjust to your liking
class PlayerController:
    def __init__(self, speed=5, speed2=15, record_inputs=False, max_queue_size=1000):
        self.force_amount = speed
        self.force_amount2 = speed2

        self.total_pitch = 0.0
        self.total_yaw = 0.0

        # Recording settings
        self.record_inputs = record_inputs
        self.input_queue = deque(maxlen=max_queue_size)

    def record_input(self, dt, keys, mouse):

        self.input_queue.append({
            "dt": dt,
            "keys": keys.copy(),
            "mouse_dx": mouse["dx"],
            "mouse_dy": mouse["dy"],
            "left_click": mouse["left_click"],
            "right_click": mouse["right_click"]
        })
    def get_next_input(self):
        """
        Remove and return the oldest input in the queue.
        Returns None if empty.
        """
        if self.input_queue:
            return self.input_queue.popleft()
        return []

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

    def mouse_controller(self, dt):

        x, y = mouse.get_position()

        dx = x - CENTER_X
        dy = y - CENTER_Y

        sensitivity = 0.001

        # Apply rotation
        self.total_yaw -= dx * sensitivity
        self.total_pitch += dy * sensitivity

        pitch_q = Quaternion.axis_angle(Vector3(1, 0, 0), self.total_pitch)
        yaw_q = Quaternion.axis_angle(Vector3(0, 1, 0), self.total_yaw)

        self.parent.quaternion = yaw_q * pitch_q

        # Record mouse input snapshot
        mouse_data = {
            "dx": dx,
            "dy": dy,
            "left_click": mouse.is_pressed("left"),
            "right_click": mouse.is_pressed("right")
        }


        mouse.move(CENTER_X, CENTER_Y)

        return mouse_data

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

        if keyboard.is_pressed('space'):
            pressed_keys.append('space')
            self.parent.Rigidbody.velocity += Vector3(0, self.force_amount2, 0) * dt

        if keyboard.is_pressed('left shift'):
            pressed_keys.append('left shift')
            self.parent.Rigidbody.velocity += Vector3(0, -self.force_amount2, 0) * dt
        return pressed_keys

    def Start(self):
        self.render = self.parent.Camera.render
        mouse.move(CENTER_X, CENTER_Y)
        self.total_pitch = self.parent.rotation.x
        self.total_yaw = self.parent.rotation.y

    def Update(self, dt):
        pressed_keys = self.keyboard_controller(dt)
        mouse_data = self.mouse_controller(dt)

        self.record_input(dt, pressed_keys, mouse_data)


