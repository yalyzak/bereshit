import keyboard
from bereshit import Vector3D

class CamController:
    def __init__(self):
        self.force_amount = 0.02

    def OnTriggerEnter(self, other):
        if other.obj.name == "wall":
            print(f"{self.parent.name} touched the wall")


        elif other.obj.name == "goal":
            print(f"{self.parent.name} reached the goal")

    def keyboard_controller(self):
        # Movement controls
        if keyboard.is_pressed('w'):
            self.parent.position += Vector3D(0, 0, self.force_amount)
        if keyboard.is_pressed('s'):
            self.parent.position += Vector3D(0, 0, -self.force_amount)
        if keyboard.is_pressed('d'):
            self.parent.position += Vector3D(self.force_amount, 0, 0)
        if keyboard.is_pressed('a'):
            self.parent.position += Vector3D(-self.force_amount, 0, 0)
        if keyboard.is_pressed('space'):
            self.parent.position += Vector3D(0, self.force_amount, 0)
        if keyboard.is_pressed('left shift'):
            self.parent.position += Vector3D(0, -self.force_amount, 0)
        # Rotation controls
        rot_speed = 0.1  # Adjust rotation speed as needed
        cam_rot = self.parent.rotation  # Assuming camera.rotation is a Vector3D

        if keyboard.is_pressed('left'):
            cam_rot.y -= rot_speed
        if keyboard.is_pressed('right'):
            cam_rot.y += rot_speed
        if keyboard.is_pressed('up'):
            cam_rot.x -= rot_speed
        if keyboard.is_pressed('down'):
            cam_rot.x += rot_speed

        self.parent.set_rotation(cam_rot)

    def main(self):
        self.keyboard_controller()