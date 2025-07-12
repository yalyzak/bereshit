from bereshit import Vector3
class Trap:

    def OnTriggerEnter(self, other):
        if other.parent.name == "car":
            print("💀 You fell into the trap!")
            other.parent.reset_to_default()
