import numpy as np

from bereshit.Vector3 import Vector3


def skew(v: Vector3):
    """Returns the 3x3 skew-symmetric (cross-product) matrix of a Vector3."""
    x, y, z = v.x, v.y, v.z
    return np.array([
        [0, -z,  y],
        [z,  0, -x],
        [-y, x,  0]
    ])


class HingeJoint:
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

    def __init__(self, body_b, axis: Vector3, friction_coefficient=0.0,
                 anchor=None):
        """
        Parameters
        ----------
        body_b : Object
            The other body to connect to (body_a is the owner).
        axis : Vector3
            Hinge axis expressed in body_a's local space.
        friction_coefficient : float
            Coulomb friction coefficient that resists rotation around the
            hinge axis.  0 = frictionless hinge.
        anchor : Vector3 or None
            World-space anchor point.  If None, the midpoint between the
            two bodies is used.
        """
        self.body_b = body_b
        self.axis_local = axis.normalized()          # hinge axis in A's local frame
        self.friction_coefficient = friction_coefficient
        self._anchor_override = anchor               # optional explicit anchor

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #
    def attach(self, owner_object):
        """Called by the component system when the joint is added."""
        self.body_a = owner_object

        self.a = self.body_a.get_component("Rigidbody")
        self.b = self.body_b.get_component("Rigidbody")

        # World-space anchor (default = midpoint)
        if self._anchor_override is not None:
            world_anchor = self._anchor_override
        else:
            world_anchor = (self.body_a.position + self.body_b.position) * 0.5

        # Store anchor in each body's local frame
        self.local_anchor_a = self.body_a.quaternion.inverse().rotate(
            world_anchor - self.body_a.position
        )
        self.local_anchor_b = self.body_b.quaternion.inverse().rotate(
            world_anchor - self.body_b.position
        )

        # Store the initial relative rotation so we can measure drift
        self.initial_rel_rot = (
            self.body_a.quaternion.inverse() * self.body_b.quaternion
        )

        return "joint"

    # ------------------------------------------------------------------ #
    #  Main solver — called every physics sub-iteration
    # ------------------------------------------------------------------ #
    def solve(self, dt):
        # Integrate pending forces into velocity (same pattern as FixedJoint)
        if not self.a.isKinematic:
            self.a.velocity += (self.a.force / self.a.mass) * dt
            self.a.force = Vector3()
        if not self.b.isKinematic:
            self.b.velocity += (self.b.force / self.b.mass) * dt
            self.b.force = Vector3()

        self.solve_linear(dt)
        self.solve_angular(dt)

    # ------------------------------------------------------------------ #
    #  1) Linear constraint — keep the anchors together
    # ------------------------------------------------------------------ #
    def _inv_mass(self, rb):
        """Return 0 for kinematic bodies, else rb.inv_mass."""
        return 0.0 if rb.isKinematic else rb.inv_mass

    def solve_linear(self, dt):
        a, b = self.a, self.b

        IinvA = a.Iinv_world()   # already returns zeros for kinematic
        IinvB = b.Iinv_world()

        inv_mA = self._inv_mass(a)
        inv_mB = self._inv_mass(b)

        # World-space lever arms
        rA = self.body_a.quaternion.rotate(self.local_anchor_a)
        rB = self.body_b.quaternion.rotate(self.local_anchor_b)

        # Velocities at the anchor points
        vA = a.velocity + a.angular_velocity.cross(rA)
        vB = b.velocity + b.angular_velocity.cross(rB)
        dv = vB - vA

        # Baumgarte position correction (bias)
        world_anchor_A = self.body_a.position + rA
        world_anchor_B = self.body_b.position + rB
        pos_error = world_anchor_B - world_anchor_A

        beta = 0.2
        bias = pos_error * (beta / dt)

        # Effective mass matrix  K = (1/mA + 1/mB)*I + [rA]x * IinvA * [rA]x^T
        #                                             + [rB]x * IinvB * [rB]x^T
        inv_mass = inv_mA + inv_mB
        rA_skew = skew(rA)
        rB_skew = skew(rB)

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
            a.angular_velocity -= Vector3.from_np(
                IinvA @ np.cross(rA.to_np(), impulse_np)
            )
        if not b.isKinematic:
            b.angular_velocity += Vector3.from_np(
                IinvB @ np.cross(rB.to_np(), impulse_np)
            )

    # ------------------------------------------------------------------ #
    #  2) Angular constraint — only allow rotation around the hinge axis
    # ------------------------------------------------------------------ #
    def solve_angular(self, dt):
        a, b = self.a, self.b

        IinvA = a.Iinv_world()
        IinvB = b.Iinv_world()

        # Hinge axis in world space (attached to body A)
        axis_world = self.body_a.quaternion.rotate(self.axis_local).normalized()

        # Build two axes perpendicular to the hinge axis
        t1 = self._perp(axis_world)
        t2 = axis_world.cross(t1).normalized()

        # Relative angular velocity
        rel_w = b.angular_velocity - a.angular_velocity

        # Construct the 2-row Jacobian: J = [t1^T; t2^T]
        J = np.array([t1.to_np(), t2.to_np()])          # (2, 3)

        # Effective mass:  K_ang = J * (IinvA + IinvB) * J^T   (2x2)
        K_full = IinvA + IinvB                            # (3, 3)
        K_ang = J @ K_full @ J.T                          # (2, 2)

        # Velocity error projected onto the two constrained directions
        vel_error = J @ rel_w.to_np()                     # (2,)

        # Baumgarte angular correction
        q_rel = self.body_a.quaternion.inverse() * self.body_b.quaternion
        q_error = q_rel * self.initial_rel_rot.inverse()

        # The vector part of the error quaternion gives the rotation error
        err_vec = Vector3(q_error.x, q_error.y, q_error.z)
        if q_error.w < 0:
            err_vec = err_vec * -1
        ang_error = err_vec * 2.0

        # Project the angular error onto the two constrained axes
        beta = 0.2
        bias = J @ ang_error.to_np() * (beta / dt)       # (2,)

        # Solve for the 2D impulse
        ang_impulse_2d = -np.linalg.solve(K_ang, vel_error + bias)  # (2,)

        # Expand back to 3D: impulse = t1 * lambda1 + t2 * lambda2
        ang_impulse_np = J.T @ ang_impulse_2d             # (3,)

        # Apply angular impulse
        if not a.isKinematic:
            a.angular_velocity -= Vector3.from_np(IinvA @ ang_impulse_np)
        if not b.isKinematic:
            b.angular_velocity += Vector3.from_np(IinvB @ ang_impulse_np)

        # ---------------------------------------------------------------
        #  3) Hinge-axis friction (optional)
        # ---------------------------------------------------------------
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
    def _perp(axis):
        """Return an arbitrary unit vector perpendicular to *axis*."""
        a = axis.to_np()
        # Pick the cardinal axis least aligned with `a`
        if abs(a[0]) < 0.9:
            candidate = np.array([1.0, 0.0, 0.0])
        else:
            candidate = np.array([0.0, 1.0, 0.0])
        perp = np.cross(a, candidate)
        perp /= np.linalg.norm(perp)
        return Vector3.from_np(perp)
