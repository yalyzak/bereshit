import math
import keyboard

from bereshit import Object, Vector3, Core, Camera, FixedJoint, HingeJoint, BoxCollider, Rigidbody, Quaternion
from bereshit.addons.essentials import FPS_cam, CamController


class HingeJointTest:
    def __init__(self, hinge):
        self.hinge = hinge
    # def Start(self):
    #     self.hinge.body_a.Rigidbody.angular_velocity += Vector3(16,0,0)
    def Update(self, dt):
        # print(self.parent.name)
        print("angular_velocity :", self.hinge.body_a.Rigidbody.angular_velocity)
        # print("quaternion :", self.hinge.body_a.quaternion)
        # print("axis_world :", self.hinge.axis_world)


class ServoController:
    def __init__(self, axis):
        self.target_angle = 0.0

        self.kP = 0.5  # converts angle error → desired speed
        self.kD = 1  # constant speed
        self.max_speed = 6.54  # rad/s
        self.max_torque = 10.32  # strength of motor
        self._axis = axis
        self.max_rotation = 180
        self.min_rotation = -180
        self.target_angle = 0.0
        self.time = 1
        self.time2 = 0
        self.did = False
        self.direction = -1

    def Update(self, dt):
        if keyboard.is_pressed('w'):
            self.target_angle -= 120.0 * dt
        if keyboard.is_pressed('s'):
            self.target_angle += 120.0 * dt
        self.target_angle = max(min(self.target_angle, self.max_rotation), self.min_rotation)
        self.fix(dt)

    def fix(self, dt):
        rb = self.parent.Rigidbody

        # Extract angle around axis from quaternion (NOT Euler angles - they have singularities at 90°)
        q = self.parent.quaternion

        # Ensure we're on the short path
        # if q.w < 0:
        #     q = q.

        q_vec = Vector3(q.x, q.y, q.z)
        current_angle = math.degrees(2.0 * math.atan2(q_vec.dot(self._axis), q.w))

        # Normalize to [-180, 180]
        current_angle = (current_angle + 180) % 360 - 180

        error = self.target_angle - current_angle

        # normalize angle error to [-180, 180]
        error = (error + 180) % 360 - 180

        angular_velocity = rb.angular_velocity.dot(self._axis)

        # torque (this creates acceleration naturally)
        torque = self.kP * error - self.kD * angular_velocity

        # clamp torque
        torque = max(min(torque, self.max_torque), -self.max_torque)

        impulse = torque * dt * self._axis
        self.apply_angular_impulse(impulse)
        # self.clamp_speed()

    def clamp_rotation(self):
        self.parent.quaternion = max(min(self.parent.quaternion.to_euler(), self.max_rotation), -self.max_rotation)

    def clamp_speed(self):
        self.parent.Rigidbody.angular_velocity.z = max(min(self.parent.Rigidbody.angular_velocity.z, self.max_speed),
                                                       -self.max_speed)

    def apply_angular_impulse(self, impulse: Vector3):
        rb = self.parent.Rigidbody
        delta_w = rb.Iinv_world() @ impulse.to_np()
        rb.angular_velocity += Vector3.from_np(delta_w)


class ServoControllerTest(ServoController):
    def Update(self, dt):
        self._axis = self.parent.get_component(HingeJoint).axis_world.normalized()
        # print(self._axis)
        # if self.time2 > 4 and self.time2 < 5:
        #     self.did = True
        #     self.parent.Rigidbody.angular_velocity = Vector3()
        if self.time2 > 3:
            # self.parent.get_component(HingeJoint).d = 5
            # self.parent.get_component(HingeJoint).beta = 0

            if abs(self.target_angle) > 100 and self.time > 1:

                self.direction *= -1
                self.time = 0
            self.time += dt
            self.target_angle += 50.0 * dt * self.direction

            self.target_angle = max(min(self.target_angle, self.max_rotation), self.min_rotation)
            self.fix(dt)
        self.time2 += dt



cam = Object(position=Vector3(0, 0, -8)).add_component(Camera(), CamController())

axis1 = Vector3(0, 0, 1)

mount = Object(name="mount", size=Vector3(1, 1, 1), position=Vector3(0, -2, 0)).add_component(
    Rigidbody(isKinematic=True), BoxCollider())

servo = Object(name="servo", ).add_component(BoxCollider(), Rigidbody(angular_velocity=Vector3(0, 0, 0)),
                                             HingeJoint(mount, axis1, anchor=Vector3(0, -2, 0)), ServoControllerTest(axis1))

mount.quaternion *= Quaternion.euler(Vector3(90, 0, 0))
# servo.quaternion *= Quaternion.euler(Vector3(90, 0, 0))

# servo.position = Vector3(0,-2, -2)
test = Object(name ="test1", size=Vector3(0, 0, 0)).add_component(HingeJointTest(servo.get_component(HingeJoint)))


axis2 = Vector3(0, 1, 0)

mount2 = Object(name="mount2", size=Vector3(1, 1, 1), position=Vector3(5, -2, 0)).add_component(
    Rigidbody(isKinematic=True), BoxCollider())

servo2 = Object(name="servo2", position=Vector3(5,-2,-2)).add_component(BoxCollider(), Rigidbody(angular_velocity=Vector3(0, 0, 0)),
                                             HingeJoint(mount2, axis2, anchor=Vector3(5, -2, 0)), ServoControllerTest(axis2))


test2 = Object(name="test2", size=Vector3(0, 0, 0)).add_component(HingeJointTest(servo2.get_component(HingeJoint)))


Core.run([cam, mount, servo, mount2, servo2, test], gravity=Vector3(), physics_epochs=2)
