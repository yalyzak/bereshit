import numpy as np

from bereshit import Vector3


class FixJoint:
    def __init__(self, other_object):
        """
        other_object: the Object you want to fix to.
        """
        self.other_object = other_object
        self.bodyA = None
        self.bodyB = other_object.get_component("Rigidbody")

        # Compute initial offset
        self.local_offset = None

    def attach(self, owner_object):
        self.bodyA = owner_object.get_component("Rigidbody")
        A = self.bodyA
        B = self.bodyB

        """
        Called when this component is attached to an object.
        """

        if A is None or B is None:
            raise ValueError("FixJoint requires both objects to have rigidbodies")
        if B.isKinematic:
            raise ValueError("can not joint a Kinematic body")
            # World-space anchor at attach time
            # World-space anchor at attach time
        world_anchor = owner_object.position

        # --- Local offsets ---
        # Convert world anchor into each body's local space

        self.rA = A.parent.quaternion.conjugate().rotate(
            B.parent.position - A.parent.position
        )

        self.rB = B.parent.quaternion.conjugate().rotate(
            A.parent.position - B.parent.position
        )
        return "joint"

    def solve(self, dt):
        def get_rotational_resistance(r, I_inv_world):
            r_skew = np.array([[0, -r.z, r.y],
                               [r.z, 0, -r.x],
                               [-r.y, r.x, 0]])
            return r_skew @ I_inv_world @ r_skew.T

        A = self.bodyA
        B = self.bodyB

        # These represent the vector from center of mass to the joint anchor
        # If they are welded at their current positions:
        rA = self.rA  # Should be world-space offset to anchor
        rB = self.rB  # Should be world-space offset to anchor

        # 1. Calculate Relative Velocity at the anchor point
        v_anchor1 = A.velocity + A.angular_velocity.cross(rA)
        v_anchor2 = B.velocity + B.angular_velocity.cross(rB)
        relative_v = v_anchor2 - v_anchor1

        # 2. Calculate Effective Mass (K Matrix)
        inv_mass_sum = (1 / A.mass) + (1 / B.mass)
        K_linear = np.eye(3) * inv_mass_sum
        K_rotA = get_rotational_resistance(rA, A.Iinv_world())
        K_rotB = get_rotational_resistance(rB, B.Iinv_world())

        K = K_linear + K_rotA + K_rotB

        # 3. Calculate Impulse
        # We want relative_v to become 0, so impulse = K^-1 * (-relative_v)
        impulse_np = np.linalg.solve(K, -relative_v.to_np())
        impulse = Vector3.from_np(impulse_np)

        # 4. Apply Linear Impulse (Corrected accumulation)
        A.velocity -= impulse / A.mass
        B.velocity -= impulse / B.mass

        # 5. Apply Angular Impulse
        A.angular_velocity += Vector3.from_np(A.Iinv_world() @ rA.cross(impulse).to_np())
        B.angular_velocity -= Vector3.from_np(B.Iinv_world() @ rB.cross(impulse).to_np())


        # shank.Rigidbody.apply_impulse_at_point(Vector3(-1, 0, 0), Vector3(0, 1.9, 0))
        #
        # shank.Rigidbody.apply_impulse_at_point(Vector3(-1, 0, 0), Vector3(0, 3, 0))
    #
    #     inv_massA = 0.0 if A.isKinematic else 1.0 / A.mass
    #     inv_massB = 0.0 if B.isKinematic else 1.0 / B.mass
    #     if inv_massA + inv_massB == 0:
    #         return
    #
    #     # -------- LINEAR --------
    #     v_rel = B.velocity - A.velocity
    #     J = -v_rel / (inv_massA + inv_massB)
    #
    #     if not A.isKinematic:
    #         A.velocity -= J * inv_massA
    #     if not B.isKinematic:
    #         B.velocity += J * inv_massB
    #
    #     # -------- ANGULAR --------
    #     w_rel = B.angular_velocity - A.angular_velocity
    #
    #     # Inverse inertia tensors (WORLD space)
    #     IinvA = np.zeros((3, 3)) if A.isKinematic else A.Iinv_world()
    #     IinvB = np.zeros((3, 3)) if B.isKinematic else B.Iinv_world()
    #
    #     K = IinvA + IinvB
    #     if np.linalg.det(K) < 1e-8:
    #         return
    #
    #     L = -Vector3.from_np(np.linalg.solve(K, w_rel.to_np()))
    #
    #     if not A.isKinematic:
    #         A.angular_velocity -= Vector3.from_np(IinvA @ L.to_np())
    #     if not B.isKinematic:
    #         B.angular_velocity += Vector3.from_np(IinvB @ L.to_np())

    # def solve(self, dt):
    #     A = self.bodyA
    #     B = self.bodyB
    #
    #     self.rA = A.parent.quaternion.conjugate().rotate(
    #         B.parent.position - A.parent.position
    #     )
    #
    #     self.rB = B.parent.quaternion.conjugate().rotate(
    #         A.parent.position - B.parent.position
    #     )
    #
    #     vA = A.velocity + A.angular_velocity.cross(self.rA)
    #     vB = B.velocity + B.angular_velocity.cross(self.rB)
    #
    #     relative_vel = vB - vA
    #     normal = relative_vel.normalized()
    #
    #     rn1 = self.rA.cross(normal)
    #
    #     rn2 = self.rB.cross(normal)
    #
    #     if not A.isKinematic:
    #         term1 = (Vector3.from_np(A.Iinv_world() @ rn1.to_np())).cross(self.rA)
    #     else:
    #         term1 = Vector3(0, 0, 0)
    #
    #     if not B.isKinematic:
    #         term2 = (Vector3.from_np(B.Iinv_world() @ rn2.to_np())).cross(self.rB)
    #     else:
    #         term2 = Vector3(0, 0, 0)
    #
    #     k_linear = (0 if A.isKinematic else 1 / A.mass) + (0 if B.isKinematic else 1 / B.mass)
    #     k_angular = normal.dot(term1 + term2)
    #     inv_eff_mass = k_linear + k_angular
    #     j = -relative_vel / inv_eff_mass
    #
    #
    #     # K = 1/A.mass + normal.dot(Vector3.from_np(A.Iinv_world() @ (self.rA.cross(normal)).to_np()).cross(self.rA))
    #     # J = -relative_vel / K
    #     # impulse_vec1 = J * normal
    #     # K = 1 / B.mass + normal.dot(Vector3.from_np(B.Iinv_world() @ (self.rB.cross(normal)).to_np()).cross(self.rB))
    #     # J = -relative_vel / K
    #     impulse_vec = j * normal
    #     if not A.isKinematic:
    #         A.velocity -= impulse_vec / A.mass
    #         # torque_impulse = self.rA.cross(impulse_vec)
    #         # ang_impulse = Vector3.from_np(A.Iinv_world() @ torque_impulse.to_np())
    #         # A.angular_velocity += ang_impulse
    #     if not B.isKinematic:
    #         B.velocity -= impulse_vec / B.mass
    #         # torque_impulse = self.rB.cross(impulse_vec)
    #         # ang_impulse = Vector3.from_np(B.Iinv_world() @ torque_impulse.to_np())
    #         # B.angular_velocity -= ang_impulse
    # def solve(self, dt):
    #
    #     A = self.bodyA
    #     B = self.bodyB
    #
    #     inv_mA = 0.0 if A.isKinematic else 1.0 / A.mass
    #     inv_mB = 0.0 if B.isKinematic else 1.0 / B.mass
    #
    #     if inv_mA + inv_mB == 0:
    #         return
    #
    #     # World-space anchor offsets
    #     rA = A.parent.quaternion.rotate(self.local_offsetA)
    #     rB = B.parent.quaternion.rotate(self.local_offsetB)
    #
    #     # Velocities at anchor points
    #     vA = A.velocity + A.angular_velocity.cross(rA)
    #     vB = B.velocity + B.angular_velocity.cross(rB)
    #
    #     # Relative velocity
    #     v_rel = vB - vA
    #
    #     # Inverse inertia tensors (world space)
    #     IinvA = self.bodyA.Iinv_world()
    #     IinvB = self.bodyB.Iinv_world()
    #
    #     # --- Effective mass (vector form) ---
    #     K = inv_mA + inv_mB
    #
    #     if not A.isKinematic:
    #         K += rA.cross(Vector3.from_np(IinvA @ rA.cross(Vector3(1, 0, 0)).to_np())).x
    #         K += rA.cross(Vector3.from_np(IinvA @ rA.cross(Vector3(0, 1, 0)).to_np())).y
    #         K += rA.cross(Vector3.from_np(IinvA @ rA.cross(Vector3(0, 0, 1)).to_np())).z
    #
    #     if not B.isKinematic:
    #         K += rB.cross(Vector3.from_np(IinvB @ rA.cross(Vector3(1, 0, 0)).to_np())).x
    #         K += rB.cross(Vector3.from_np(IinvB @ rA.cross(Vector3(0, 1, 0)).to_np())).y
    #         K += rB.cross(Vector3.from_np(IinvB @ rA.cross(Vector3(0, 0, 1)).to_np())).z
    #
    #     if K == 0:
    #         return
    #
    #     # Impulse
    #     J = -v_rel / K
    #
    #     # Apply linear impulse
    #     if not A.isKinematic:
    #         A.velocity -= J * inv_mA
    #     if not B.isKinematic:
    #         B.velocity += J * inv_mB
    #
    #     # Apply angular impulse
    #     if not A.isKinematic:
    #         A.angular_velocity -= Vector3.from_np(IinvA @ rA.cross(J).to_np())
    #     if not B.isKinematic:
    #         B.angular_velocity += Vector3.from_np(IinvB @ rB.cross(J).to_np())
