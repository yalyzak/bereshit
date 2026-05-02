import numpy as np
from bereshit.Vector3 import Vector3
from bereshit.Joint import Joint


class FixedJoint(Joint):

    def __cast_anchor_default(self):
        self.world_anchor = (self.body_a.position + self.body_b.position) * 0.5

    def solve_linear(self, dt):
        a, b = self.rbA, self.rbB

        IinvA = a.Iinv_world()  # already returns zeros for kinematic
        IinvB = b.Iinv_world()

        inv_mA = a.invMass
        inv_mB = b.invMass

        # World-space lever arms
        rA = self.body_a.quaternion.conjugate().rotate(self.local_anchor_a)
        rB = self.body_b.quaternion.conjugate().rotate(self.local_anchor_b)

        # Velocities at the anchor points
        vA = a.velocity + a.angular_velocity.cross(-rA)
        vB = b.velocity + b.angular_velocity.cross(-rB)
        dv = vB - vA

        # Baumgarte position correction (bias)
        world_anchor_A = self.body_a.position + rA
        world_anchor_B = self.body_b.position + rB
        pos_error = world_anchor_B - world_anchor_A

        bias = pos_error * (self.beta / dt)

        # Effective mass matrix  K = (1/mA + 1/mB)*I + [rA]x * IinvA * [rA]x^T
        #                                             + [rB]x * IinvB * [rB]x^T
        inv_mass = inv_mA + inv_mB
        rA_skew = rA.skew()
        rB_skew = rB.skew()

        K = (
                inv_mass * np.identity(3)
                + rA_skew @ IinvA @ rA_skew.T
                + rB_skew @ IinvB @ rB_skew.T
        )

        # Solve  K * impulse = -(dv + bias)
        impulse_np = -np.linalg.solve(K, (dv + bias).to_np())
        impulse = Vector3.from_np(impulse_np)

        # Apply linear impulse
        if not a.isKinematic:
            a.velocity -= impulse * inv_mA
        if not b.isKinematic:
            b.velocity += impulse * inv_mB

        # Apply angular impulse from the lever arms
        if not a.isKinematic:
            a.angular_velocity += Vector3.from_np(
                IinvA @ np.cross(rA.to_np(), impulse_np)
            )
        if not b.isKinematic:
            b.angular_velocity -= Vector3.from_np(
                IinvB @ np.cross(rB.to_np(), impulse_np)
            )

    def solve_angular(self, dt):
        IA = self.rbA.Iinv_world()
        IB = self.rbB.Iinv_world()

        q_rel = (
                self.body_a.quaternion.inverse() *
                self.body_b.quaternion
        )

        q_error = q_rel * self.initial_rel_rot.inverse()

        error = Vector3(q_error.x, q_error.y, q_error.z)
        if q_error.w < 0:
            error = error * -1

        angular_error = error * 2.0

        bias = angular_error * (self.beta / dt)

        rel_w = self.rbB.angular_velocity - self.rbA.angular_velocity

        K = IA + IB

        impulse = -np.linalg.solve(K, (rel_w + bias).to_np())
        if not self.rbA.isKinematic:
            self.rbA.angular_velocity -= Vector3.from_np(IA @ impulse)
        self.rbB.angular_velocity += Vector3.from_np(IB @ impulse)
