from bereshit.Vector3 import Vector3
from bereshit.Physics import Physics
import warnings


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
        hit = Physics.Raycast(self.body_a.position, -(self.body_b.position - self.body_a.position), self.body_a.Collider).point

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