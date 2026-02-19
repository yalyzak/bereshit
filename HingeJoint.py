import numpy as np

from bereshit.Vector3 import Vector3

def skew(v: Vector3):
    x, y, z = v.x, v.y, v.z
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

class HingeJoint:
    def __init__(self, body_b, axis: Vector3, friction_coefficient=0):
        self.body_b = body_b
        self.axis_local = axis.normalized()  # hinge axis in A's local space
        self.axis_local_inv = Vector3(1,1,1) - axis.normalized()
        self.axis_local_inv += self.axis_local * friction_coefficient

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
        self.solve_impulse()


    # --------------------------------------------------
    # Proper 3D linear constraint with angular couplingd
    # --------------------------------------------------
    def solve_impulse(self):
        a = self.a
        b = self.b

        # ---- World anchors (use Transform quaternion) ----
        rA = self.body_a.quaternion.rotate(self.local_anchor_a)
        rB = self.body_b.quaternion.rotate(self.local_anchor_b)

        # ---- Velocities at anchor ----
        vA = a.velocity + a.angular_velocity.cross(rA)
        vB = b.velocity + b.angular_velocity.cross(rB)

        relative_velocity = vB - vA

        # =========================================================
        # 1) Linear Constraint (with angular coupling)
        # =========================================================

        IinvA = a.Iinv_world()
        IinvB = b.Iinv_world()

        inv_mass = a.inv_mass + b.inv_mass

        rA_skew = skew(rA)
        rB_skew = skew(rB)

        K = (
                inv_mass * np.identity(3)
                - rA_skew @ IinvA @ rA_skew.T
                - rB_skew @ IinvB @ rB_skew.T
        )

        impulse_np = -np.linalg.solve(K, relative_velocity.to_np())
        impulse = Vector3.from_np(impulse_np)

        # Apply linear
        a.velocity += impulse * a.inv_mass
        b.velocity += impulse * b.inv_mass

        # Apply angular
        # a.angular_velocity -= Vector3.from_np(IinvA @ rA.cross(impulse).to_np())
        # b.angular_velocity += Vector3.from_np(IinvB @ rB.cross(impulse).to_np())

        # =========================================================
        # 2) Angular Hinge Constraint
        # =========================================================

        axis_world = self.body_a.quaternion.rotate(self.axis_local).normalized()

        relative_w = b.angular_velocity - a.angular_velocity
        perp = relative_w - axis_world * relative_w.dot(axis_world)

        K_ang = IinvA + IinvB

        ang_impulse_np = -np.linalg.solve(K_ang, perp.to_np())
        ang_impulse = Vector3.from_np(ang_impulse_np)

        # a.angular_velocity -= Vector3.from_np(IinvA @ ang_impulse.to_np())
        # b.angular_velocity += Vector3.from_np(IinvB @ ang_impulse.to_np())




