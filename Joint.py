import numpy as np

from bereshit.Vector3 import Vector3
from bereshit.Physics import Physics
import warnings
from numba import njit

class Joint:
    solve3x3Array = np.empty(3, dtype=float)
    solve2x2Array = np.empty(2, dtype=float)

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
        self.inv_mass_array = np.eye(3)
        self.ang_impulse = Vector3()


    def attach(self, owner_object):
        """Called by the component system when the joint is added."""
        self.body_a = owner_object

        self.rbA = self.body_a.get_component("Rigidbody")
        self.rbB = self.body_b.get_component("Rigidbody")

        if self.world_anchor is None:
            self.cast_anchor()

        return "Joint"

    def cast_anchor(self):
        hit = Physics.Raycast(self.body_a.position, (self.body_b.position - self.body_a.position), self.body_b.Collider).point

        if hit is not None:
            self.world_anchor = hit
        else:
            self.__cast_anchor_default()  # fall back

        # Store the initial relative rotation so we can measure drift
        self.initial_rel_rot = (self.body_a.quaternion.inverse() * self.body_b.quaternion)

        self.local_anchor_a = self.body_a.quaternion.rotate_conjugated(self.world_anchor - self.body_a.position)
        self.local_anchor_b = self.body_b.quaternion.rotate_conjugated(self.world_anchor - self.body_b.position)


    def __cast_anchor_default(self):
        self.world_anchor = self.body_b.position

    def solve(self, dt):
        # Integrate pending forces into velocity (same pattern as FixedJoint)
        if not self.rbA.isKinematic:
            self.rbA.velocity += (self.rbA.force / self.rbA.mass) * dt
            self.rbA.force.Zero()
        if not self.rbB.isKinematic:
            self.rbB.velocity += (self.rbB.force / self.rbB.mass) * dt
            self.rbB.force.Zero()
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
    def solve3x3(K, b):
        Joint.__solve3x3(K, b, Joint.solve3x3Array)
        return Joint.solve3x3Array

    @staticmethod
    @njit
    def __solve3x3(K, b, array):
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

        array[0] = m00 * b[0] + m01 * b[1] + m02 * b[2]
        array[1] = m10 * b[0] + m11 * b[1] + m12 * b[2]
        array[2] = m20 * b[0] + m21 * b[1] + m22 * b[2]

    @staticmethod
    def solve2x2(K, b):
        Joint.__solve2x2(K, b, Joint.solve2x2Array)
        return Joint.solve2x2Array

    @staticmethod
    @njit
    def __solve2x2(K, b, array):
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
        array[0] = (e * b[0] - c * b[1]) * inv_det
        array[1] = (-d * b[0] + a * b[1]) * inv_det



    @staticmethod
    @njit
    def solve_sym_3x3(K, b):
        a00, a01, a02 = K[0]
        a10, a11, a12 = K[1]
        a20, a21, a22 = K[2]

        det = (
                a00 * (a11 * a22 - a12 * a21)
                - a01 * (a10 * a22 - a12 * a20)
                + a02 * (a10 * a21 - a11 * a20)
        )

        inv_det = 1.0 / det

        # Cofactor inverse
        i00 = (a11 * a22 - a12 * a21) * inv_det
        i01 = (a02 * a21 - a01 * a22) * inv_det
        i02 = (a01 * a12 - a02 * a11) * inv_det

        i10 = (a12 * a20 - a10 * a22) * inv_det
        i11 = (a00 * a22 - a02 * a20) * inv_det
        i12 = (a02 * a10 - a00 * a12) * inv_det

        i20 = (a10 * a21 - a11 * a20) * inv_det
        i21 = (a01 * a20 - a00 * a21) * inv_det
        i22 = (a00 * a11 - a01 * a10) * inv_det

        return np.array([
            i00 * b[0] + i01 * b[1] + i02 * b[2],
            i10 * b[0] + i11 * b[1] + i12 * b[2],
            i20 * b[0] + i21 * b[1] + i22 * b[2],
        ])