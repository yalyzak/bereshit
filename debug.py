import time

from bereshit import Quaternion,Vector3,Mesh_rander,rotate_vector_old,rotate_vector_quaternion
import  mouse

CENTER_X = 960
CENTER_Y = 540
class debug:
    def __init__(self):
        self.total_pitch = 0.0
        self.total_yaw = 0.0
        self.state = 0
        self.start_time = 0
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
    def Update(self):
        self.change()
        # self.s()
        # 1. Compute relative position from pivot to object


        # self.parent.quaternion *= Quaternion.euler(Vector3(0,0,0.01))

    # #
    def Start(self):
        self.parent.Rigidbody.angular_velocity += Vector3(3,1,10)

    #     self.parent.quaternion *= Quaternion.euler(Vector3(0,0,30))
        # print(self.parent.rotation)
        self.parent.Rigidbody.velocity += Vector3(3,1,5)