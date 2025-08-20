import time

import bereshit
from bereshit import Quaternion,Vector3,Mesh_rander,rotate_vector_old,rotate_vector_quaternion
import  mouse

CENTER_X = 960
CENTER_Y = 540
class debug:
    def __init__(self,other):
        self.total_pitch = 0.0
        self.total_yaw = 0.0
        self.state = 0
        self.start_time = 0
        self.max = 0
        self.dt = 0
        self.other = other
    def change(self):
        current_time = time.perf_counter()

        if self.state == 0:
            self.parent.add_component(Mesh_rander(shape="empty"))

            self.start_time = current_time
            self.state = 1

        # elif self.state == 1:
        #     if current_time - self.start_time >= 5.0:
        #         self.parent.add_component(Mesh_rander(shape="box"))
        #
        #         self.state = 0  # Done
    # def s(self):
    #     relative = self.parent.position - pivot
    #
    #     # 2. Rotate that vector
    #     rotation = Quaternion.euler(Vector3(0, 0, 0.01))  # Rotate around Z axis
    #     rotated_relative = rotate_vector_quaternion(relative, rotation)
    #
    #     # 3. Set new position to maintain orbit
    #     self.parent.position = pivot + rotated_relative
    #
    #     # 4. Optionally rotate the object itself (e.g., to face same direction)
    #     self.parent.quaternion = rotation * self.parent.quaternion  # Global rotation
    def des(self):
        # self.parent = bereshit.Object()
        self.parent.world.add_child(bereshit.Object(size=(10,10,10),name="äsd"))
        # self.parent.world.children.append(bereshit.Object(size=(10,10,10)))
        print(self.parent.world.children)
    def Update(self,dt):
        e = 0.5 * self.parent.Rigidbody.mass * self.parent.Rigidbody.velocity.magnitude() ** 2 + self.parent.Rigidbody.mass * 9.8 * abs(
            self.parent.position.y)
        e2 = 0.5 * self.other.Rigidbody.mass * self.other.Rigidbody.velocity.magnitude() ** 2 + self.other.Rigidbody.mass * 9.8 * abs(
            self.other.position.y)

        print(e + e2)
    # def Start(self):
    #     self.max = self.parent.position.y
    #     # self.parent.Rigidbody.angular_velocity += Vector3(3,1,10)
    #     self.des()
    #
    # #     self.parent.quaternion *= Quaternion.euler(Vector3(0,0,30))
    #     # print(self.parent.rotation)
    #     self.parent.Rigidbody.velocity += Vector3(0,0,2)