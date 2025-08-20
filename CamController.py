import time

import keyboard
import Core
from bereshit import Vector3,rotate_vector_quaternion
from bereshit import Quaternion

class CamController:
    def __init__(self,speed=5):
        self.force_amount = speed
        self.force_amount2 = speed

    def keyboard_controller(self,dt):
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
        if keyboard.is_pressed('w'):

            self.parent.position += rotate_vector_quaternion(Vector3(0, 0, self.force_amount),self.parent.quaternion) * dt

        if keyboard.is_pressed('s'):
            self.parent.position += rotate_vector_quaternion(Vector3(0, 0, -self.force_amount), self.parent.quaternion) * dt



        if keyboard.is_pressed('d'):
            self.parent.position += rotate_vector_quaternion(Vector3(-self.force_amount, 0, 0), self.parent.quaternion) * dt


        if keyboard.is_pressed('a'):
            self.parent.position += rotate_vector_quaternion(Vector3(self.force_amount, 0, 0), self.parent.quaternion) * dt


        if keyboard.is_pressed('space'):
            self.parent.position += Vector3(0, self.force_amount2, 0) * dt

        if keyboard.is_pressed('left shift'):
            self.parent.position += Vector3(0, -self.force_amount2, 0) * dt




    def Update(self,dt):
        self.keyboard_controller(dt)

        # # Get the player's current position
        # player_pos = self.player.position
        #
        # # Calculate new camera position: 10 cm behind player
        # new_cam_pos = player_pos + Vector3(0,1.0,- 0.2)
        #
        # # Update this camera's transform
        # self.parent.position = new_cam_pos
