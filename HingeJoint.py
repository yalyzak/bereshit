import math

import numpy as np

from bereshit.Vector3 import Vector3
from bereshit.Joint import Joint
from bereshit.Physics import Physics
from bereshit.Vector2 import Vector2


class HingeJoint(Joint):
    """
    Revolute (hinge) joint — connects two rigid bodies and allows
    rotation around a single axis only.

    Constraints enforced:
      1. Linear  (3 DOF removed) — anchor points on A and B stay together
      2. Angular (2 DOF removed) — relative rotation perpendicular to the
         hinge axis is eliminated
      3. (Optional) Friction    — resists rotation around the hinge axis

    Total: 6 DOF → 1 DOF (free rotation around the hinge axis).
    """

    def __init__(self, body_b, axis, friction_coefficient=0.0, anchor=None, max_rotation=180, min_rotation=-180, beta=0.2):
        super().__init__(body_b, anchor, beta)
        self.axis_world = Vector3()
        self.axis_local = axis.normalized()  # hinge axis in A's local frame
        self.friction_coefficient = friction_coefficient
        self.max_rotation = max_rotation
        self.min_rotation = min_rotation
        self.K_ang = np.empty((2, 2), dtype=float)
        self.vel_error = np.empty(2, dtype=float) # (2,)
        self.bias = np.empty(2, dtype=float) # (2,)



    def __cast_anchor_default(self):
        self.world_anchor = self.body_b.position

    def solve(self, dt):
        super().solve(dt)
        self.axis_world = self.body_b.quaternion.rotate_conjugated(self.axis_local).normalized()

    def solve_linear(self, dt):
        a, b = self.rbA, self.rbB

        IinvA = a.Iinv_world()  # already returns zeros for kinematic
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

        K = Joint.build_effective_mass_matrix(inv_mass, rA, rB, IinvA, IinvA, self.K)

        # Solve  K * impulse = -(dv + bias)
        impulse_np = -Joint.solve3x3(K, (dv + bias).to_np())
        impulse = Vector3.from_np(impulse_np)

        # 15

        # Apply linear impulse
        if not a.isKinematic:
            a.velocity -= impulse * inv_mA
        if not b.isKinematic:
            b.velocity += impulse * inv_mB

        # Apply angular impulse from the lever arms
        if not a.isKinematic:
            a.angular_velocity += rA.cross(impulse).MatrixMultiplication(IinvA)

        if not b.isKinematic:
            b.angular_velocity -= rB.cross(impulse).MatrixMultiplication(IinvB)

    def solve_angular(self, dt):
        a, b = self.rbA, self.rbB

        IinvA = a.Iinv_world()
        IinvB = b.Iinv_world()

        # Hinge axis in world space (attached to body A)
        axis_world = self.body_a.quaternion.rotate_conjugated(self.axis_local).normalized()

        # Build two axes perpendicular to the hinge axis
        t1 = self.__perp(axis_world)
        t2 = axis_world.cross(t1).normalized()

        # Relative angular velocity
        rel_w = b.angular_velocity - a.angular_velocity

        # Construct the 2-row Jacobian: J = [t1^T; t2^T]
        # Effective mass:  K_ang = J * (IinvA + IinvB) * J^T   (2x2)
        K_full = IinvA + IinvB  # (3, 3)

        Kt1 = t1.MatrixMultiplication(K_full)
        Kt2 = t2.MatrixMultiplication(K_full)

        # J @ K_full @ J.T
        self.K_ang[0] = t1.dot(Kt1), t1.dot(Kt2)
        self.K_ang[1] = t2.dot(Kt1), t2.dot(Kt2)

        # Velocity error projected onto the two constrained directions
        self.vel_error[0] = t1.dot(rel_w)
        self.vel_error[1] = t2.dot(rel_w)

        # Baumgarte angular correction
        q_rel = self.body_a.quaternion.inverse() * self.body_b.quaternion

        self.clamp_rotation(q_rel, IinvA, IinvB, a, b)

        q_error = q_rel * self.initial_rel_rot.inverse()

        # The vector part of the error quaternion gives the rotation error
        err_vec = Vector3(q_error.x, q_error.y, q_error.z)
        if q_error.w < 0:
            err_vec = err_vec * -1
        ang_error = err_vec * 2.0

        # Project the angular error onto the two constrained axes
        self.bias[0] = t1.dot(ang_error) * (self.beta / dt)
        self.bias[1] = t2.dot(ang_error) * (self.beta / dt)

        # Solve for the 2D impulse
        ang_impulse_np = -Joint.solve2x2(self.K_ang, self.vel_error + self.bias)  # (2,)
        ang_impulse = Vector2.from_np(ang_impulse_np)

        # Expand back to 3D: impulse = t1 * lambda1 + t2 * lambda2
        self.ang_impulse.x = Vector2(t1.x, t2.x).dot(ang_impulse)
        self.ang_impulse.y = Vector2(t1.y, t2.y).dot(ang_impulse)
        self.ang_impulse.z = Vector2(t1.z, t2.z).dot(ang_impulse)


        # Apply angular impulse
        if not a.isKinematic:
            a.angular_velocity -= self.ang_impulse.MatrixMultiplication(IinvA)
        if not b.isKinematic:
            b.angular_velocity += self.ang_impulse.MatrixMultiplication(IinvB)

        # ---------------------------------------------------------------
        #  3) Hinge-axis friction (optional)
        # ---------------------------------------------------------------
        # self.friction()

    def clamp_rotation(self, q_rel, IinvA, IinvB, a, b):
        axis_world = self.body_a.quaternion.rotate(self.axis_local).normalized()  # should be body b i think

        # extract angle from quaternion
        angle = 2 * math.atan2(
            Vector3(q_rel.x, q_rel.y, q_rel.z).dot(axis_world),
            q_rel.w
        )

        angle = math.degrees(angle)
        angle = (angle + 180) % 360 - 180
        clamped_angle = max(self.min_rotation, min(angle, self.max_rotation))
        error = angle - clamped_angle
        if abs(error) > 0.001:
            correction_speed = error * 0.0001  # stiffness (tune this)

            impulse = axis_world * (-correction_speed)

            if not a.isKinematic:
                a.angular_velocity -= Vector3.from_np(IinvA @ impulse.to_np())
            if not b.isKinematic:
                b.angular_velocity += Vector3.from_np(IinvB @ impulse.to_np())

    def friction(self, rel_w, axis_world, K_full, ang_impulse_2d, IinvA, IinvB, a, b):
        if self.friction_coefficient > 0:
            # Relative angular velocity around the hinge axis
            w_hinge = rel_w.dot(axis_world)

            # Effective mass for a single-axis constraint
            k_hinge = axis_world.to_np() @ K_full @ axis_world.to_np()

            # Impulse to stop rotation around the hinge
            friction_impulse = -w_hinge / k_hinge

            # Clamp by Coulomb friction
            max_friction = self.friction_coefficient * (
                    np.linalg.norm(ang_impulse_2d) + 1e-8
            )
            friction_impulse = max(-max_friction,
                                   min(friction_impulse, max_friction))

            f_imp = axis_world.to_np() * friction_impulse

            if not a.isKinematic:
                a.angular_velocity -= Vector3.from_np(IinvA @ f_imp)
            if not b.isKinematic:
                b.angular_velocity += Vector3.from_np(IinvB @ f_imp)

    # ------------------------------------------------------------------ #
    #  Helper — find a vector perpendicular to the given axis
    # ------------------------------------------------------------------ #
    @staticmethod
    def __perp(axis):
        """Return an arbitrary unit vector perpendicular to axis."""
        x, y, z = axis.x, axis.y, axis.z

        # Pick axis least aligned with input
        if abs(x) < 0.9:
            helper = Vector3(1.0, 0.0, 0.0)
        else:
            helper = Vector3(0.0, 1.0, 0.0)

        return axis.cross(helper).normalized()

    @staticmethod
    def __add_K(K, r, I):
        rx, ry, rz = r.x, r.y, r.z

        i00, i01, i02 = I[0]
        i10, i11, i12 = I[1]
        i20, i21, i22 = I[2]

        K[0, 0] += ry * ry * i22 - ry * rz * (i12 + i21) + rz * rz * i11
        K[0, 1] += -rx * ry * i22 + rx * rz * i21 + ry * rz * i02 - rz * rz * i01
        K[0, 2] += rx * ry * i12 - rx * rz * i11 - ry * ry * i02 + ry * rz * i01

        K[1, 0] += -rx * ry * i22 + rx * rz * i12 + ry * rz * i20 - rz * rz * i10
        K[1, 1] += rx * rx * i22 - rx * rz * (i02 + i20) + rz * rz * i00
        K[1, 2] += -rx * rx * i12 + rx * ry * i02 + rx * rz * i10 - ry * rz * i00

        K[2, 0] += rx * ry * i21 - rx * rz * i11 - ry * ry * i20 + ry * rz * i10
        K[2, 1] += -rx * rx * i21 + rx * ry * i01 + rx * rz * i20 - ry * rz * i00
        K[2, 2] += rx * rx * i11 - rx * ry * (i01 + i10) + ry * ry * i00

    @staticmethod
    def __overwrite_K(K, r, I):
        rx, ry, rz = r.x, r.y, r.z

        i00, i01, i02 = I[0]
        i10, i11, i12 = I[1]
        i20, i21, i22 = I[2]

        K[0, 0] += ry * ry * i22 - ry * rz * (i12 + i21) + rz * rz * i11
        K[0, 1] = -rx * ry * i22 + rx * rz * i21 + ry * rz * i02 - rz * rz * i01
        K[0, 2] = rx * ry * i12 - rx * rz * i11 - ry * ry * i02 + ry * rz * i01

        K[1, 0] = -rx * ry * i22 + rx * rz * i12 + ry * rz * i20 - rz * rz * i10
        K[1, 1] += rx * rx * i22 - rx * rz * (i02 + i20) + rz * rz * i00
        K[1, 2] = -rx * rx * i12 + rx * ry * i02 + rx * rz * i10 - ry * rz * i00

        K[2, 0] = rx * ry * i21 - rx * rz * i11 - ry * ry * i20 + ry * rz * i10
        K[2, 1] = -rx * rx * i21 + rx * ry * i01 + rx * rz * i20 - ry * rz * i00