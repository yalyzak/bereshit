from bereshit import Vector3


class FixJoint:
    def __init__(self, other_object):
        """
        other_object: the Object you want to fix to.
        """
        self.other_object = other_object
        self.bodyA = None  # Will be filled in at attach time
        self.bodyB = other_object.get_component("Rigidbody")

        # Compute initial offset
        self.local_offset = None

    def attach(self, owner_object):
        """
        Called when this component is attached to an object.
        """
        self.bodyA = owner_object.get_component("Rigidbody")
        if self.bodyA is None or self.bodyB is None:
            raise ValueError("FixJoint requires both objects to have rigidbodies")
        if self.bodyB.isKinematic:
            raise ValueError("can not joint a Kinematic body")
        self.local_offset = self.bodyB.parent.position - self.bodyA.parent.position
        self.anchor_world = self.bodyA.parent.position + self.local_offset
        self.defaultA = self.bodyA.parent.quaternion
        self.defaultB = self.bodyB.parent.quaternion
        return "joint"

    def solve(self, dt):

        A = self.bodyA
        B = self.bodyB

        # Vector from A to B
        r = B.parent.position - A.parent.position
        dist = r.magnitude()

        if dist == 0:
            return

        r_hat = r / dist

        # Integrate forces -> velocities
        if not A.isKinematic:
            A.velocity += (A.force / A.mass) * dt
        if not B.isKinematic:
            B.velocity += (B.force / B.mass) * dt

        A.force = Vector3()
        B.force = Vector3()

        # Relative velocity
        v_rel = B.velocity - A.velocity

        # Radial relative velocity (constraint violation)
        v_radial = v_rel.dot(r_hat)

        # If already satisfied, do nothing
        if v_radial == 0:
            return

        # Inverse masses
        invMassA = 0.0 if A.isKinematic else 1.0 / A.mass
        invMassB = 0.0 if B.isKinematic else 1.0 / B.mass
        totalInvMass = invMassA + invMassB

        if totalInvMass == 0:
            return

        # Constraint impulse (velocity-level)
        impulse = r_hat * (v_radial / totalInvMass)

        # Apply equal and opposite responses
        if not A.isKinematic:
            A.velocity += impulse * invMassA

        if not B.isKinematic:
            B.velocity -= impulse * invMassB

