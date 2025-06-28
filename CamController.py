import keyboard
from bereshit import Vector3D

class CamController:
    def __init__(self):
        self.force_amount = 10

    def OnTriggerEnter(self, other):
        if other.obj.name == "wall":
            print(f"{self.parent.name} touched the wall")


        elif other.obj.name == "goal":
            print(f"{self.parent.name} reached the goal")

    def keyboard_controller(self):
        # Update the key state


        if keyboard.is_pressed('o'):
            self.parent.position += Vector3D(0, 0, self.force_amount)
        if keyboard.is_pressed('l'):
            self.parent.position += Vector3D(0, 0, -self.force_amount)
        if keyboard.is_pressed(';'):
            self.parent.position += Vector3D(self.force_amount, 0, 0)
        if keyboard.is_pressed('k'):
            self.parent.position += Vector3D(-self.force_amount, 0, 0)
        if keyboard.is_pressed('space'):
            self.parent.position += Vector3D(0, self.force_amount, 0)
        if keyboard.is_pressed('left shift'):
            self.parent.position += Vector3D(0, -self.force_amount, 0)


    def main(self):
        self.keyboard_controller()