import numpy as np

from bereshit import Quaternion
from bereshit.Vector3 import Vector3



class Rigidbody:
    _friction_table = {
        ("Steel", "Concrete"): 0.6,
        ("Rubber", "Concrete"): 0.9,
        ("Rubber", "Steel"): 0.8,
        ("Wood", "Ice"): 0.04,
        ("Steel", "Steel"): 0.5,
        ("floor", "Steel"): 0.2,
        ("Steel", "Steel"): 0.2,
        # ("Steel", "Ice"): 0.1,
        # add more as needed
    }
    _default_friction = 0.6

    def __init__(self, obj=None, mass=1.0, size=Vector3(1, 1, 1), position=Vector3(0, 0, 0),
                 center_of_mass=Vector3(0, 0, 0), velocity=None, angular_velocity=None, force=None,
                 isKinematic=False, useGravity=True, drag=0.98, friction_coefficient=0.6, restitution=0.6,COM=None, Freeze_Rotation=None):
        self.mass = mass
        self.material = ""
        self.drag = drag
        self.obj = obj
        self.energy = 0
        self.restitution = restitution
        self.friction_coefficient = friction_coefficient if friction_coefficient is not None else self._default_friction
        self.center_of_mass = center_of_mass if center_of_mass else position
        self.velocity = velocity or Vector3(0, 0, 0)
        self.acceleration = Vector3()
        self.angular_acceleration = Vector3(0, 0, 0)
        self.torque = Vector3()
        self.force = force or Vector3(0, 0, 0)
        self.isKinematic = isKinematic
        self.forward = Vector3()
        self.Freeze_Rotation = Freeze_Rotation or Vector3(0, 0, 0)
        self.useGravity = useGravity

        if angular_velocity is None:
            angular_velocity = Vector3(0, 0, 0)
        self.angular_velocity = angular_velocity

        self.normal_force = Vector3()
        self._COM = COM

    def Iinv_world(self):
        # This should only be called for dynamic bodies
        if self.isKinematic:
            return np.zeros((3, 3))

        # 1. Get the local inverse inertia tensor (3x3 matrix)
        # Ensure this is the INVERSE, not the base inertia
        I_inv_body = self.inverse_inertia

        # 2. Get the rotation matrix
        # If using a quaternion:
        R = self.parent.quaternion.to_matrix3()
        # R = Quaternion().to_matrix3()

        # 3. Transform to world space: R * I_inv * R_transpose
        return R @ I_inv_body @ R.T
    def integrat(self, dt):
        # === 4) INTEGRATION PHASE ===
        # 4.1) Linear acceleration & velocity:
        self.acceleration = self.force / self.mass

        # 4.2) Angular acceleration & velocity (component‐wise):
        # I_world_inv = Iinv_world(rb)
        # rb.angular_acceleration = I_world_inv @ rb.torque
        self.angular_acceleration = Vector3(
            self.torque.x / self.inertia.x if self.inertia.x != 0 else 0,
            self.torque.y / self.inertia.y if self.inertia.y != 0 else 0,
            self.torque.z / self.inertia.z if self.inertia.z != 0 else 0
        )
        # 4.3) Integrate rotation

        ang_disp = self.angular_velocity * dt \
                   + 0.5 * self.angular_acceleration * dt * dt
        self.angular_velocity += self.angular_acceleration * dt
        # self.angular_velocity *= 0.8
        w = self.angular_velocity
        mag = w.magnitude()
        # if mag > 0:
        #     self.angular_velocity -= w.normalized() * (0.1 * mag * mag * dt / (1/60))
        self.parent.quaternion *= Quaternion.euler_radians(ang_disp)

        self.parent.position += self.velocity * dt \
                         + 0.5 * self.acceleration * dt * dt

        self.velocity += self.acceleration * dt

        self.force = Vector3(0, 0, 0)
        self.torque = Vector3(0, 0, 0)

        self.angular_acceleration = Vector3(0, 0, 0)
        self.torque = Vector3(0, 0, 0)

    def apply_impulse_at_point(rb,velocity ,impulse_point_world):
        """
        rb                     : Rigidbody
        impulse_point_world    : Vector3 (world space)
        impulse_dir            : Vector3 (world space, does NOT need to be normalized)
        delta_v_desired        : float (velocity change along impulse_dir)
        """

        if rb.isKinematic:
            return
        impulse_dir = velocity.normalized()
        delta_v_desired = velocity.magnitude()
        # --- Normalize impulse direction ---
        n = impulse_dir.normalized()

        # --- Vector from COM to impulse point (world space) ---
        r = impulse_point_world - rb.parent.position

        inv_mass = 1.0 / rb.mass

        # --- World inverse inertia tensor ---
        Iinv = rb.Iinv_world()  # 3x3 numpy matrix

        # --- Effective mass calculation ---
        rn = r.cross(n)
        Iinv_rn = Vector3.from_np(Iinv @ rn.to_np())
        angular_term = n.dot(Iinv_rn.cross(r))

        k = inv_mass + angular_term

        if k <= 1e-8:
            return  # avoid division by zero / singular config

        # --- Solve impulse magnitude ---
        j = delta_v_desired / k

        # --- Impulse vector ---
        J = n * j

        # --- Apply linear impulse ---
        rb.velocity += J * inv_mass

        # --- Apply angular impulse ---
        rb.angular_velocity += Vector3.from_np(
            Iinv @ r.cross(J).to_np()
        )

    def _get_friction(self, other_rb):
        """
        Returns the friction coefficient for the pair of materials.
        """

        if not other_rb:
            return self.friction_coefficient
        if self.friction_coefficient != Rigidbody._default_friction or other_rb.friction_coefficient != Rigidbody._default_friction:
            return min(self.friction_coefficient, other_rb.friction_coefficient)
        mat1 = self.material
        mat2 = other_rb.material
        key = (mat1, mat2)
        rev_key = (mat2, mat1)

        if key in Rigidbody._friction_table:
            return Rigidbody._friction_table[key]
        elif rev_key in Rigidbody._friction_table:
            return Rigidbody._friction_table[rev_key]
        else:
            return Rigidbody._default_friction

    def AddForce(self, force, ContactPoint=None):
        # Linear force always contributes directly to acceleration
        self.force += force

        if ContactPoint is not None:
            ContactPoint = self.parent.position + ContactPoint
            # r is the lever arm (vector from center of mass to contact point)
            r = self.parent.position - ContactPoint
            # torque = r × F
            self.torque += r.cross(force)

    def attach(self, owner_object):
        # self.size = owner_object.size
        # self.position = owner_object.position
        if self._COM is None:
            hx = owner_object.size.x
            hy = owner_object.size.y
            hz = owner_object.size.z
            self.inertia = Vector3(
                (1 / 12) * self.mass * (hy ** 2 + hz ** 2),
                (1 / 12) * self.mass * (hx ** 2 + hz ** 2),
                (1 / 12) * self.mass * (hy ** 2 + hx ** 2)
            )
            self.inertia_tensor = np.array([
                [self.inertia.x, 0.0, 0.0],
                [0.0, self.inertia.y, 0.0],
                [0.0, 0.0, self.inertia.z]
            ])
        else:
            def box_inertia_vector(mass, hx, hy, hz, com_offset):
                """
                Returns the inertia as a Vector3 (Ixx, Iyy, Izz)
                for a solid box with dimensions hx, hy, hz,
                shifted by a COM offset using the parallel-axis theorem.

                com_offset = Vector3(dx, dy, dz)
                """

                # === Base inertia at geometric center ===
                Ixx = (1 / 12) * mass * (hy * hy + hz * hz)
                Iyy = (1 / 12) * mass * (hx * hx + hz * hz)
                Izz = (1 / 12) * mass * (hx * hx + hy * hy)

                # === COM offset ===
                dx, dy, dz = com_offset.x, com_offset.y, com_offset.z

                # parallel-axis correction (diagonal terms)
                # d^2 - dx^2  → contributes to Ixx
                # d^2 - dy^2  → contributes to Iyy
                # d^2 - dz^2  → contributes to Izz
                d2 = dx * dx + dy * dy + dz * dz

                Ixx += mass * (d2 - dx * dx)
                Iyy += mass * (d2 - dy * dy)
                Izz += mass * (d2 - dz * dz)

                return Vector3(Ixx, Iyy, Izz)

            hx = owner_object.size.x
            hy = owner_object.size.y
            hz = owner_object.size.z



            self.inertia = box_inertia_vector(self.mass, hx, hy, hz, self._COM)
        self.center_of_mass = owner_object.position
        self.obj = owner_object
        self.material = owner_object.material.kind
        self.forward = owner_object.quaternion.rotate(owner_object.position)
        EPSILON = 1e-8  # Small value to avoid division by zero





        def safe_inverse(value):
            return 1.0 / value if abs(value) > EPSILON else 0.0

        self.inverse_inertia = np.diag([
            safe_inverse(self.inertia.x),
            safe_inverse(self.inertia.y),
            safe_inverse(self.inertia.z)
        ])