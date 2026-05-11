import numpy as np

from bereshit.Vector3 import Vector3
from bereshit.Physics import Physics
import warnings
from numba import njit

class Joint:
    def __init__(self, body_b, anchor=None, beta=0.2):
        self.local_anchor_b = None
        self.local_anchor_a = None
        self.initial_rel_rot = None
        self.body_a = None
        self.rbB = None
        self.rbA = None
        self.body_b = body_b
        self.world_anchor = anchor
        self.beta = beta

    def attach(self, owner_object):
        """Called by the component system when the joint is added."""
        self.body_a = owner_object

        self.rbA = self.body_a.get_component("Rigidbody")
        self.rbB = self.body_b.get_component("Rigidbody")

        if self.world_anchor is None:
            self.cast_anchor()

        # Store the initial relative rotation so we can measure drift
        self.initial_rel_rot = (self.body_a.quaternion.inverse() * self.body_b.quaternion)

        self.local_anchor_a = self.body_a.quaternion.inverse().rotate(self.world_anchor - self.body_a.position)
        self.local_anchor_b = self.body_b.quaternion.inverse().rotate(self.world_anchor - self.body_b.position)

        return "joint"

    def cast_anchor(self):
        hit = Physics.Raycast(self.body_a.position, (self.body_b.position - self.body_a.position), self.body_b.Collider).point

        if hit is not None:
            self.world_anchor = hit
        else:
            self.__cast_anchor_default()  # fall back

    def __cast_anchor_default(self):
        self.world_anchor = self.body_b.position

    def solve(self, dt):
        # Integrate pending forces into velocity (same pattern as FixedJoint)
        if not self.rbA.isKinematic:
            self.rbA.velocity += (self.rbA.force / self.rbA.mass) * dt
            self.rbA.force = Vector3()
        if not self.rbB.isKinematic:
            self.rbB.velocity += (self.rbB.force / self.rbB.mass) * dt
            self.rbB.force = Vector3()
        self.solve_linear(dt)
        self.solve_angular(dt)

    def solve_linear(self, dt):
        warnings.warn(
            f"{self.__class__.__name__}.solve_linear(dt) was not implemented",
            UserWarning,
            stacklevel=2
        )

    def solve_angular(self, dt):
        warnings.warn(
            f"{self.__class__.__name__}.solve_angular(dt) was not implemented",
            UserWarning,
            stacklevel=2
        )

    @staticmethod
    @njit
    def solve3x3(K, b):
        """
        Solve Kx = b for a 3x3 matrix using Cramer's rule.

        Parameters
        ----------
        K : (3,3) ndarray
        b : (3,) ndarray

        Returns
        -------
        x : (3,) ndarray
        """

        a = K[0, 0]
        b1 = K[0, 1]
        c = K[0, 2]

        d = K[1, 0]
        e = K[1, 1]
        f = K[1, 2]

        g = K[2, 0]
        h = K[2, 1]
        i = K[2, 2]

        # determinant
        det = (
                a * (e * i - f * h)
                - b1 * (d * i - f * g)
                + c * (d * h - e * g)
        )

        if abs(det) < 1e-12:
            raise ValueError("Singular matrix")

        inv_det = 1.0 / det

        # inverse matrix entries
        m00 = (e * i - f * h) * inv_det
        m01 = (c * h - b1 * i) * inv_det
        m02 = (b1 * f - c * e) * inv_det

        m10 = (f * g - d * i) * inv_det
        m11 = (a * i - c * g) * inv_det
        m12 = (c * d - a * f) * inv_det

        m20 = (d * h - e * g) * inv_det
        m21 = (b1 * g - a * h) * inv_det
        m22 = (a * e - b1 * d) * inv_det

        x0 = m00 * b[0] + m01 * b[1] + m02 * b[2]
        x1 = m10 * b[0] + m11 * b[1] + m12 * b[2]
        x2 = m20 * b[0] + m21 * b[1] + m22 * b[2]

        return np.array([x0, x1, x2])

    @staticmethod
    @njit
    def solve2x2(K, b):
        """
        Solve Kx = b for a 2x2 matrix.

        Parameters
        ----------
        K : (2,2) ndarray
        b : (2,) ndarray

        Returns
        -------
        x : (2,) ndarray
        """

        a = K[0, 0]
        c = K[0, 1]

        d = K[1, 0]
        e = K[1, 1]

        det = a * e - c * d

        if abs(det) < 1e-12:
            raise ValueError("Singular matrix")

        inv_det = 1.0 / det

        # inverse(K) * b
        x0 = (e * b[0] - c * b[1]) * inv_det
        x1 = (-d * b[0] + a * b[1]) * inv_det

        return np.array([x0, x1])