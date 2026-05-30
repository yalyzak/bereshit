from bereshit.Vector3 import Vector3
from bereshit.Joint import Joint


class FixedJoint(Joint):

    def solve_linear(self, dt):
        a, b = self.rbA, self.rbB

        IinvA = a.Iinv_world()
        IinvB = b.Iinv_world()

        inv_mA = a.invMass
        inv_mB = b.invMass

        # World-space lever arms
        rA = self.body_a.quaternion.rotate_conjugated(self.local_anchor_a)
        rB = self.body_b.quaternion.rotate_conjugated(self.local_anchor_b)

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

        K = Joint.build_effective_mass_matrix(inv_mass, rA, rB, IinvA, IinvB, self.K)

        # Solve  K * impulse = -(dv + bias)
        impulse_np = -Joint.solve3x3(K, (dv + bias).to_np())
        impulse = Vector3.from_np(impulse_np)

        if not a.isKinematic:
            a.velocity -= impulse * inv_mA
            a.angular_velocity += rA.cross(impulse).MatrixMultiplication(IinvA)

        if not b.isKinematic:
            b.velocity += impulse * inv_mB
            b.angular_velocity -= rB.cross(impulse).MatrixMultiplication(IinvB)

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

        impulse = -Joint.solve3x3(K, (rel_w + bias).to_np())
        impulse = Vector3.from_np(impulse)

        if not self.rbA.isKinematic:
            self.rbA.angular_velocity -= impulse.MatrixMultiplication(IA)
        self.rbB.angular_velocity += impulse.MatrixMultiplication(IB)
