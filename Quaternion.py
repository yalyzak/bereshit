import math
import numpy as np
from numba import njit
from bereshit.Vector3 import Vector3


class Quaternion:
    __slots__ = ("x", "y", "z", "w")
    def __init__(self, x=0, y=0, z=0, w=1):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __repr__(self):
        return f"Quaternion({self.x}, {self.y}, {self.z}, {self.w})"

    def __copy__(self):
        return Quaternion(self.x, self.y, self.z, self.w)

    def __neg__(self):
        return Quaternion(-self.x, -self.y, -self.z, -self.w)

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)
        elif isinstance(other, (list, tuple)) and len(other) == 4:
            return Quaternion(self.x + other[0], self.y + other[1], self.z + other[2], self.w + other[3])
        raise TypeError(f"Unsupported type for addition: {type(other)}")

    def __iadd__(self, other):
        result = self + other
        self.x, self.y, self.z, self.w = result.x, result.y, result.z, result.w
        return self

    def __sub__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)
        elif isinstance(other, (list, tuple)) and len(other) == 4:
            return Quaternion(self.x - other[0], self.y - other[1], self.z - other[2], self.w - other[3])
        raise TypeError(f"Unsupported type for subtraction: {type(other)}")

    def __mul__(self, other):
        w = self.w
        return Quaternion(
            w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        )


    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(self.x / other, self.y / other, self.z / other, self.w / other)
        elif isinstance(other, Quaternion):
            return self * other.inverse()
        raise TypeError("Unsupported type for division")

    def conjugate(self):
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    def norm(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2 + self.w ** 2)

    def normalized(self):
        n = self.norm()
        if n == 0:
            return Quaternion(0, 0, 0, 1)
        return self / n

    def inverse(self):
        norm_sq = self.x ** 2 + self.y ** 2 + self.z ** 2 + self.w ** 2
        return Quaternion(
            -self.x / norm_sq,
            -self.y / norm_sq,
            -self.z / norm_sq,
            self.w / norm_sq
        )

    def to_euler(self):
        """
        Converts the quaternion to Euler angles (roll, pitch, yaw) in radians.
        Convention: ZYX (yaw-pitch-roll)
        Returns:
            (roll, pitch, yaw)
        """
        x, y, z, w = self.x, self.y, self.z, self.w

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return Vector3(math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

    def look_rotation(forward: "Vector3", up: "Vector3") -> "Quaternion":
        """
        Builds a quaternion that rotates the +Z axis to align with `forward`,
        keeping `up` as close as possible to the given up vector.
        """
        f = forward.normalized()
        u = up.normalized()
        r = u.cross(f).normalized()
        u = f.cross(r)  # recompute to ensure orthogonality

        # Construct a rotation matrix from r, u, f
        m00, m01, m02 = r.x, u.x, f.x
        m10, m11, m12 = r.y, u.y, f.y
        m20, m21, m22 = r.z, u.z, f.z

        # Convert rotation matrix → quaternion
        trace = m00 + m11 + m22

        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (m21 - m12) / s
            y = (m02 - m20) / s
            z = (m10 - m01) / s
        elif (m00 > m11) and (m00 > m22):
            s = math.sqrt(1.0 + m00 - m11 - m22) * 2
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = math.sqrt(1.0 + m11 - m00 - m22) * 2
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = math.sqrt(1.0 + m22 - m00 - m11) * 2
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s

        return Quaternion(x, y, z, w)

    @classmethod
    def euler_radians(cls, vec3):  # vec3 = Vector3(roll, pitch, yaw)

        roll = vec3.x
        pitch = vec3.y
        yaw = vec3.z

        c1 = math.cos(yaw / 2)
        s1 = math.sin(yaw / 2)
        c2 = math.cos(pitch / 2)
        s2 = math.sin(pitch / 2)
        c3 = math.cos(roll / 2)
        s3 = math.sin(roll / 2)

        w = c1 * c2 * c3 + s1 * s2 * s3
        x = c1 * c2 * s3 - s1 * s2 * c3
        y = c1 * s2 * c3 + s1 * c2 * s3
        z = s1 * c2 * c3 - c1 * s2 * s3

        return cls(x, y, z, w)

    @classmethod
    def euler(cls, vec3):  # vec3 = Vector3(roll, pitch, yaw)
        roll = math.radians(vec3.x)
        pitch = math.radians(vec3.y)
        yaw = math.radians(vec3.z)

        c1 = math.cos(yaw / 2)
        s1 = math.sin(yaw / 2)
        c2 = math.cos(pitch / 2)
        s2 = math.sin(pitch / 2)
        c3 = math.cos(roll / 2)
        s3 = math.sin(roll / 2)

        w = c1 * c2 * c3 + s1 * s2 * s3
        x = c1 * c2 * s3 - s1 * s2 * c3
        y = c1 * s2 * c3 + s1 * c2 * s3
        z = s1 * c2 * c3 - c1 * s2 * s3

        return cls(x, y, z, w)

    @staticmethod
    def from_axis_angle(axis: Vector3, angle: float):
        half = angle * 0.5
        s = math.sin(half)
        return Quaternion(
            math.cos(half),
            axis.x * s,
            axis.y * s,
            axis.z * s
        )

    def look_rotation(forward: "Vector3", up: "Vector3") -> "Quaternion":
        """
        Builds a quaternion that rotates the +Z axis to align with `forward`,
        keeping `up` as close as possible to the given up vector.
        """
        f = forward.normalized()
        u = up.normalized()
        r = u.cross(f).normalized()
        u = f.cross(r)  # recompute to ensure orthogonality

        # Construct a rotation matrix from r, u, f
        m00, m01, m02 = r.x, u.x, f.x
        m10, m11, m12 = r.y, u.y, f.y
        m20, m21, m22 = r.z, u.z, f.z

        # Convert rotation matrix → quaternion
        trace = m00 + m11 + m22

        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (m21 - m12) / s
            y = (m02 - m20) / s
            z = (m10 - m01) / s
        elif (m00 > m11) and (m00 > m22):
            s = math.sqrt(1.0 + m00 - m11 - m22) * 2
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = math.sqrt(1.0 + m11 - m00 - m22) * 2
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = math.sqrt(1.0 + m22 - m00 - m11) * 2
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s

        return Quaternion(x, y, z, w)

    @staticmethod
    def axis_angle(axis, angle_rad):
        """Create a quaternion representing a rotation of angle_rad around 'axis' (Vector3)."""
        half_angle = angle_rad * 0.5
        sin_half = math.sin(half_angle)
        axis_n = axis.normalized()  # Make sure your Vector3 has this method

        return Quaternion(
            axis_n.x * sin_half,
            axis_n.y * sin_half,
            axis_n.z * sin_half,
            math.cos(half_angle)
        )

    def to_axis_angle(self):
        """Convert this quaternion to (axis: Vector3, angle: float)"""
        if abs(self.w) > 1:
            self = self.normalized()

        angle = 2 * math.acos(self.w)
        s = math.sqrt(1 - self.w * self.w)

        if s < 1e-6:
            # If s is too small, return arbitrary axis
            return Vector3(1, 0, 0), 0.0
        else:
            return Vector3(self.x / s, self.y / s, self.z / s), angle

    def to_matrix3(self, out):
        # if out is None:
        #     out = np.empty((3, 3), dtype=np.float64)

        x = self.x
        y = self.y
        z = self.z
        w = self.w

        xx = 2.0 * x * x
        yy = 2.0 * y * y
        zz = 2.0 * z * z

        xy = 2.0 * x * y
        xz = 2.0 * x * z
        yz = 2.0 * y * z

        wx = 2.0 * w * x
        wy = 2.0 * w * y
        wz = 2.0 * w * z

        out[0, 0] = 1.0 - yy - zz
        out[0, 1] = xy - wz
        out[0, 2] = xz + wy

        out[1, 0] = xy + wz
        out[1, 1] = 1.0 - xx - zz
        out[1, 2] = yz - wx

        out[2, 0] = xz - wy
        out[2, 1] = yz + wx
        out[2, 2] = 1.0 - xx - yy

        return out

    def rotate(self, v: Vector3) -> Vector3:
        qx, qy, qz, qw = self.x, self.y, self.z, self.w
        vx, vy, vz = v.x, v.y, v.z

        # t = 2 * cross(q.xyz, v)
        tx = 2 * (qy * vz - qz * vy)
        ty = 2 * (qz * vx - qx * vz)
        tz = 2 * (qx * vy - qy * vx)

        # v' = v + qw * t + cross(q.xyz, t)
        rx = vx + qw * tx + (qy * tz - qz * ty)
        ry = vy + qw * ty + (qz * tx - qx * tz)
        rz = vz + qw * tz + (qx * ty - qy * tx)

        return Vector3(rx, ry, rz)

    def rotate_conjugated(self, v: Vector3) -> Vector3:
        qx, qy, qz, qw = self.x, self.y, self.z, self.w
        vx, vy, vz = v.x, v.y, v.z

        # t = 2 * cross(q.xyz, v)
        tx = 2 * (qy * vz - qz * vy)
        ty = 2 * (qz * vx - qx * vz)
        tz = 2 * (qx * vy - qy * vx)

        # v' = v - qw * t + cross(q.xyz, t)
        rx = vx - qw * tx + (qy * tz - qz * ty)
        ry = vy - qw * ty + (qz * tx - qx * tz)
        rz = vz - qw * tz + (qx * ty - qy * tx)

        return Vector3(rx, ry, rz)

    @staticmethod
    def from_basis(right, up, forward):
        # Normalize to be safe
        r = forward.cross(up).normalized()
        u = right.cross(forward).normalized()
        f = forward.normalized()

        # Rotation matrix (column-major)
        m00, m01, m02 = r.x, r.y, r.z
        m10, m11, m12 = u.x, u.y, u.z
        m20, m21, m22 = f.x, f.y, f.z

        trace = m00 + m11 + m22

        if trace > 0:
            s = (trace + 1.0) ** 0.5 * 2
            w = 0.25 * s
            x = (m21 - m12) / s
            y = (m02 - m20) / s
            z = (m10 - m01) / s
        elif (m00 > m11) and (m00 > m22):
            s = (1.0 + m00 - m11 - m22) ** 0.5 * 2
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = (1.0 + m11 - m00 - m22) ** 0.5 * 2
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = (1.0 + m22 - m00 - m11) ** 0.5 * 2
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s

        return Quaternion(x, y, z, w).normalized()

    def to_np(self):
        return np.array([self.x, self.y, self.z, self.w], dtype='f4')

    def to_matrix3_abs(self, m):
        x = self.x
        y = self.y
        z = self.z
        w = self.w

        xx = 2.0 * x * x
        yy = 2.0 * y * y
        zz = 2.0 * z * z

        xy = 2.0 * x * y
        xz = 2.0 * x * z
        yz = 2.0 * y * z

        wx = 2.0 * w * x
        wy = 2.0 * w * y
        wz = 2.0 * w * z

        m[0, 0] = abs(1.0 - yy - zz)
        m[0, 1] = abs(xy - wz)
        m[0, 2] = abs(xz + wy)

        m[1, 0] = abs(xy + wz)
        m[1, 1] = abs(1.0 - xx - zz)
        m[1, 2] = abs(yz - wx)

        m[2, 0] = abs(xz - wy)
        m[2, 1] = abs(yz + wx)
        m[2, 2] = abs(1.0 - xx - yy)

        return m

