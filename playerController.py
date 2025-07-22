import time

import keyboard

import render
from bereshit import Vector3
import mouse
class PlayerController:
    def __init__(self):
        self.force_amount = .03
        self.max_speed = 2
        # self.cam = cam

        self.force_amount2 = 4
    def OnCollisionStay(self, other):
        if self.parent.rigidbody.velocity.magnitude() < self.max_speed:

            if keyboard.is_pressed('w'):
                # self.parent.rigidbody.exert_force(Vector3(0, 0, self.force_amount))
                self.parent.rigidbody.velocity += Vector3(0, 0, self.force_amount)

            if keyboard.is_pressed('s'):
                self.parent.rigidbody.velocity += Vector3(0, 0, -self.force_amount)

                # self.parent.rigidbody.exert_force(Vector3(0, 0, -self.force_amount))

            if keyboard.is_pressed('d'):
                self.parent.rigidbody.velocity += Vector3(self.force_amount, 0, 0)

                # self.parent.rigidbody.exert_force(Vector3(self.force_amount, 0, 0))

            if keyboard.is_pressed('a'):
                self.parent.rigidbody.velocity += Vector3(-self.force_amount, 0, 0)

                # self.parent.rigidbody.exert_force(Vector3(-self.force_amount, 0, 0))

        if keyboard.is_pressed('space'):
            self.parent.rigidbody.velocity += Vector3(0, self.force_amount2, 0)

        if keyboard.is_pressed('left shift'):
            self.parent.rigidbody.force += Vector3(0, -self.force_amount2, 0)
            # self.parent.rigidbody.exert_force(Vector3(0, self.force_amount * 2, 0))

    def OnTriggerEnter(self, other):
        if other.obj.name == "wall":
            print(f"{self.parent.name} touched the wall")


        elif other.obj.name == "goal":
            print(f"{self.parent.name} reached the goal")

    def keyboard_controller(self):
        if keyboard.is_pressed('esc'):
            print("Paused. Release ESC to continue.")

            # Wait until ESC is released
            while keyboard.is_pressed('esc'):
                time.sleep(0.05)

            print("Now press ESC again to continue.")

            # Wait for ESC to be pressed again
            while not keyboard.is_pressed('esc'):
                time.sleep(0.05)

            # Wait for ESC to be released again before continuing
            while keyboard.is_pressed('esc'):
                time.sleep(0.05)

            print("Continuing...")

        # dx, dy = mouse.get_position()
        # # You might want relative movement:
        # dx_rel, dy_rel = mouse.get_position()
        # # mouse.move(0, 0)  # Reset mouse to origin to read relative next frame
        #
        # # Sensitivity factor (tweak to your liking)
        # sensitivity = 0.1
        #
        # # Apply torque around Y axis (yaw) and X axis (pitch)
        # yaw_torque = Vector3(0, dx_rel * sensitivity, 0)
        # pitch_torque = Vector3(dy_rel * sensitivity, 0, 0)
        # # self.parent.rigidbody.torque += yaw_torque + pitch_torque
        # self.parent.set_rotation(Vector3(dy * sensitivity,dx * sensitivity,0))
    # def start(self):
    #     self.force_amount = self.parent.rigidbody.mass * 2 * 9.8

    def Update(self):
        self.keyboard_controller()
