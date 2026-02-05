import numpy as np

from bereshit import Quaternion, World
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
    def apply_gravity(self):
        if self.useGravity:
            self.force += World.World.Gravity * self.mass
    def resolve_dynamic_collision(self, other, normal, J, r1, r2):
        """
        Applies linear and angular impulse to both dynamic bodies, factoring restitution.
        """
        rb1 = self
        rb2 = other

        impulse_vec = normal * J

        if rb1 and not rb1.isKinematic:
            rb1.velocity += impulse_vec / rb1.mass
            torque_impulse = r1.cross(impulse_vec)
            ang_impulse = Vector3.from_np(rb1.Iinv_world() @ torque_impulse.to_np())
            rb1.angular_velocity -= ang_impulse
        if rb2 and not rb2.isKinematic:
            rb2.velocity -= impulse_vec / rb2.mass
            torque_impulse = r2.cross(impulse_vec)
            ang_impulse = Vector3.from_np(rb2.Iinv_world() @ torque_impulse.to_np())
            rb2.angular_velocity += ang_impulse

    def apply_friction_impulse(self, other, normal, J, contact_point):
        """
        Applies Coulomb friction impulse based on the relative velocity,
        including angular friction effect.
        """
        rb1 = self
        rb2 = other

        v1 = rb1.velocity if rb1 and not rb1.isKinematic else Vector3(0, 0, 0)
        v2 = rb2.velocity if rb2 and not rb2.isKinematic else Vector3(0, 0, 0)
        v_rel = v1 - v2

        tangent = v_rel - normal * v_rel.dot(normal)
        tangent_length = tangent.magnitude()

        if tangent_length < 1e-6:
            return  # No significant tangential motion

        tangent = tangent.normalized()

        # Friction coefficient
        if rb1 and rb2:
            mu = rb1._get_friction(rb2)
        elif rb1:
            mu = rb1.friction_coefficient
        elif rb2:
            mu = rb2.friction_coefficient
        else:
            mu = Rigidbody._default_friction

        # Compute friction impulse scalar
        Jt_magnitude = -v_rel.dot(tangent)
        denom = 0.0
        if rb1 and not rb1.isKinematic:
            denom += 1.0 / rb1.mass
        if rb2 and not rb2.isKinematic:
            denom += 1.0 / rb2.mass

        if denom == 0.0:
            return

        Jt_magnitude /= denom
        max_friction = mu * J
        Jt_magnitude = max(-max_friction, min(Jt_magnitude, max_friction))

        Jt = tangent * Jt_magnitude

        # Apply linear and angular friction impulses
        if rb1 and not rb1.isKinematic:
            rb1.velocity += Jt / rb1.mass
            # r1 = contact_point - rb1.parent.position
            # angular_impulse1 = r1.cross(Jt)
            # rb1.angular_velocity += Vector3.from_np(rb1.inverse_inertia @ angular_impulse1.to_np())

        if rb2 and not rb2.isKinematic:
            rb2.velocity -= Jt / rb2.mass
            # r2 = contact["r2"]
            # angular_impulse2 = r2.cross(-Jt)
            # rb2.angular_velocity += Vector3.from_np(rb2.inverse_inertia @ angular_impulse2.to_np())

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
        w = self.angular_velocity
        mag = w.magnitude()
        if mag > 0:
            self.angular_velocity -= w.normalized() * (0.05  * mag * dt / (1/60))

        self.parent.quaternion *= Quaternion.euler(ang_disp)

        self.parent.position += self.velocity * dt \
                         + 0.5 * self.acceleration * dt * dt

        self.velocity += self.acceleration * dt

        self.force = Vector3(0, 0, 0)
        self.torque = Vector3(0, 0, 0)

        self.angular_acceleration = Vector3(0, 0, 0)
        self.torque = Vector3(0, 0, 0)
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
            r =  ContactPoint - self.parent.position
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

    def Iinv_world(self):
        # This should only be called for dynamic bodies
        if not self or self.isKinematic:
            return np.zeros((3, 3))

        # 1. Get the local inverse inertia tensor (3x3 matrix)
        # Ensure this is the INVERSE, not the base inertia
        I_inv_body = self.inverse_inertia

        # 2. Get the rotation matrix
        # If using a quaternion:
        R = self.parent.quaternion.to_matrix3()
        R = Quaternion().to_matrix3()

        # 3. Transform to world space: R * I_inv * R_transpose
        return R @ I_inv_body @ R.T