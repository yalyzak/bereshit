import keyboard

from bereshit import Object, Vector3, Core, Camera, FixedJoint, HingeJoint, BoxCollider, Rigidbody, Quaternion
from bereshit.addons.essentials import FPS_cam, CamController

class HingeJointTest:
    def __init__(self, hinge, arm):
        self.hinge = hinge
        self.arm = arm

    def Update(self, dt):
        # world anchor
        a = self.hinge.body_a.position + self.hinge.body_a.quaternion.rotate(self.hinge.local_anchor_a)
        b = self.hinge.body_b.position + self.hinge.body_b.quaternion.rotate(self.hinge.local_anchor_b)

        # # 🔥 1. anchor consistency
        # error = (a - b).magnitude()
        # print("Anchor error:", error)

        # 🔥 2. distance from anchor → arm center
        arm_pos = self.arm.position
        dist = (arm_pos - a).magnitude()
        print("Arm distance from hinge:", dist)

        # 🔥 3. check axis alignment (should rotate around Z)
        axis = self.hinge.axis_local.normalized()
        rel = (arm_pos - a).normalized()

        # dot should stay ~constant if rotating in plane
        dot = rel.dot(axis)
        print("Axis dot (should ~0):", dot)
class ServoController:
    def __init__(self, axis):
        self.target_angle = 0.0

        self.kP = 0.5          # converts angle error → desired speed
        self.kD = 5  # constant speed
        self.max_speed = 6.54   # rad/s
        self.max_torque = 10.32  # strength of motor
        self._axis = axis
        self.max_rotation = 90
        self.min_rotation = -90
        self.target_angle = 0.0
        self.time = 1
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

        current_angle = self.parent.quaternion.to_euler().dot(self._axis)
        error = self.target_angle - current_angle

        # normalize angle to [-180, 180] (VERY IMPORTANT)
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
        self.parent.Rigidbody.angular_velocity.z = max(min(self.parent.Rigidbody.angular_velocity.z, self.max_speed), -self.max_speed)

    def apply_angular_impulse(self, impulse: Vector3):
        rb = self.parent.Rigidbody
        delta_w = rb.Iinv_world() @ impulse.to_np()
        rb.angular_velocity += Vector3.from_np(delta_w)
class ServoControllerTest(ServoController):
    def Update(self, dt):
        if abs(self.target_angle) > 50 and self.time > 1:
            self.direction *= -1
            self.time = 0
        self.time += dt
        self.target_angle += 50.0 * dt * self.direction

        self.target_angle = max(min(self.target_angle, self.max_rotation), self.min_rotation)
        self.fix(dt)

cam = Object(position=Vector3(0,0,-8)).add_component(Camera(), FPS_cam(), CamController())

mount = Object(size=Vector3(0,0,0), position=Vector3(0,-2,0)).add_component(Rigidbody(isKinematic=True), BoxCollider())

servo = Object().add_component(BoxCollider(), Rigidbody(), HingeJoint(mount, Vector3(0,0,1), anchor=Vector3(0,0,0)), ServoControllerTest(Vector3(0,0,1)))

arm = Object(position=Vector3(0,2,0), size=Vector3(1,2,1)).add_component(BoxCollider(), Rigidbody(mass=0.03), FixedJoint(servo))

mount.quaternion *= Quaternion.euler(Vector3(90,0,0))

test = Object(size=Vector3(0,0,0)).add_component(HingeJointTest(servo.get_component(HingeJoint), arm))

Core.run([cam, mount, servo, arm, test])