import numpy as np
from bereshit.Vector3 import Vector3
from bereshit.class_type import Joint


def skew(v: Vector3):
    x, y, z = v.x, v.y, v.z
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])


class FixedJoint(Joint):
    def __init__(self, body_b):
        self.body_b = body_b


    def attach(self, owner_object):

        self.body_a = owner_object

        self.a = self.body_a.get_component("Rigidbody")
        self.b = self.body_b.get_component("Rigidbody")

        # Store initial relative transform
        self.local_offset = self.body_b.position - self.body_a.position
        self.initial_rel_rot = (
            self.body_a.quaternion.inverse() *
            self.body_b.quaternion
        )
        world_anchor = (self.body_a.position + self.body_b.position) * 0.5

        self.local_anchor_a = self.body_a.quaternion.inverse().rotate(world_anchor - self.body_a.position)
        self.local_anchor_b = self.body_b.quaternion.inverse().rotate(world_anchor - self.body_b.position)




        return "joint"


    def solve(self, dt):
        if not self.a.isKinematic:
            self.a.velocity += (self.a.force / self.a.mass) * dt
            self.a.force = Vector3()
        if not self.b.isKinematic:
            self.b.velocity += (self.b.force / self.b.mass) * dt
            self.b.force = Vector3()
        self.solve_linear(dt)
        self.solve_angular(dt)


    # --------------------------------------------------
    # Proper 3D linear constraint with angular couplingd
    # --------------------------------------------------
    def solve_linear(self, dt):
        IA = self.a.Iinv_world()
        IB = self.b.Iinv_world()

        rA = self.body_a.quaternion.conjugate().rotate(self.local_anchor_a)
        rB = self.body_b.quaternion.conjugate().rotate(self.local_anchor_b)

        # Velocity at anchors
        vA = self.a.velocity + self.a.angular_velocity.cross(-rA)
        vB = self.b.velocity + self.b.angular_velocity.cross(-rB)

        rv = vB - vA

        # Position error
        world_anchor_A = self.body_a.position + rA
        world_anchor_B = self.body_b.position + rB
        error = world_anchor_B - world_anchor_A

        beta = 0.2
        bias = error * (beta / dt)

        inv_mass = self.a.inv_mass + self.b.inv_mass

        K = (
                inv_mass * np.identity(3)
                + skew(rA) @ IA @ skew(rA).T
                + skew(rB) @ IB @ skew(rB).T
        )

        impulse = -np.linalg.solve(K, (rv + bias).to_np())

        J = Vector3.from_np(impulse)

        # Linear impulse
        if not self.a.isKinematic:
            self.a.velocity -= J * self.a.inv_mass
        self.b.velocity += J * self.b.inv_mass

        # Angular impulse
        if not self.a.isKinematic:
            self.a.angular_velocity += Vector3.from_np(
                IA @ np.cross(rA.to_np(), impulse)
            )
        self.b.angular_velocity -= Vector3.from_np(
            IB @ np.cross(rB.to_np(), impulse)
        )

    def solve_angular(self, dt):
        IA = self.a.Iinv_world()
        IB = self.b.Iinv_world()

        q_rel = (
                self.body_a.quaternion.inverse() *
                self.body_b.quaternion
        )

        q_error = q_rel * self.initial_rel_rot.inverse()

        error = Vector3(q_error.x, q_error.y, q_error.z)
        if q_error.w < 0:
            error = error * -1

        angular_error = error * 2.0

        beta = 0.2
        bias = angular_error * (beta / dt)

        rel_w = self.b.angular_velocity - self.a.angular_velocity

        K = IA + IB

        impulse = -np.linalg.solve(K, (rel_w + bias).to_np())
        if not self.a.isKinematic:
            self.a.angular_velocity -= Vector3.from_np(IA @ impulse)
        self.b.angular_velocity += Vector3.from_np(IB @ impulse)

