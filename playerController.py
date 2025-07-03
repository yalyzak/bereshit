import keyboard
from bereshit import Vector3D
#
# if keyboard.is_pressed('w'):
#     self.parent.position += Vector3D(0, 0, self.force_amount)
# if keyboard.is_pressed('s'):
#     self.parent.position += (Vector3D(0, 0, -self.force_amount))
# if keyboard.is_pressed('d'):
#     self.parent.position += (Vector3D(-self.force_amount, 0, 0))
# if keyboard.is_pressed('a'):
#     self.parent.position += (Vector3D(self.force_amount, 0, 0))
class PlayerController:
    def __init__(self):
        self.force_amount = 10

        self.force_amount2 = 50
    def OnTriggerEnter(self, other):
        if other.obj.name == "wall":
            print(f"{self.parent.name} touched the wall")


        elif other.obj.name == "goal":
            print(f"{self.parent.name} reached the goal")

    def keyboard_controller(self):
        # Update the key state


        if keyboard.is_pressed('w'):
            # self.parent.rigidbody.exert_force(Vector3D(0, 0, self.force_amount))
            self.parent.rigidbody.force +=Vector3D(0, 0, self.force_amount)
        if keyboard.is_pressed('s'):
            self.parent.rigidbody.force +=Vector3D(0, 0, -self.force_amount)

            # self.parent.rigidbody.exert_force(Vector3D(0, 0, -self.force_amount))

        if keyboard.is_pressed('d'):
            self.parent.rigidbody.force +=Vector3D(self.force_amount, 0, 0)

            # self.parent.rigidbody.exert_force(Vector3D(self.force_amount, 0, 0))

        if keyboard.is_pressed('a'):
            self.parent.rigidbody.force +=Vector3D(-self.force_amount, 0, 0)

            # self.parent.rigidbody.exert_force(Vector3D(-self.force_amount, 0, 0))

        if keyboard.is_pressed('space'):
            self.parent.rigidbody.force +=Vector3D(0, self.force_amount2, 0)
        if keyboard.is_pressed('left shift'):
            self.parent.rigidbody.force += Vector3D(0, -self.force_amount2, 0)
            # self.parent.rigidbody.exert_force(Vector3D(0, self.force_amount * 2, 0))

    # def start(self):
    #     self.force_amount = self.parent.rigidbody.mass * 2 * 9.8

    def main(self):
        # self.keyboard_controller()
        self.parent.rigidbody.force += Vector3D(10, 0, 0)