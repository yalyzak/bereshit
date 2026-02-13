from bereshit import Vector3, Object


class FixedJoint:
    def __init__(self, other_object, anchor_scale=0.0):
        self.other_object = other_object
        self.bodyA = None  # Will be filled in at attach time
        self.bodyB = other_object.get_component("Rigidbody")

        # --- choose 3 non-collinear local anchors on A ---
        d = anchor_scale
        self.localA = [
            Vector3( d, 0, 0),
            Vector3( 0, d, 0),
            Vector3( 0, 0, d),
        ]

        # --- compute matching local anchors on B ---
        self.localB = []
        self.rest_lengths = []

    def attach(self, owner_object):
        self.bodyA = owner_object.get_component("Rigidbody")
        self.gizmo = [Object(size=Vector3(0.1,0.1,0.1)) for _ in range(3*2)]
        gimos_contianer = Object(children=self.gizmo,size=Vector3(0,0,0))
        self.bodyA.parent.add_child(gimos_contianer)

        for i, la in enumerate(self.localA):
            # world position of anchor on A
            wa = (
                self.bodyA.parent.position
                + self.bodyA.parent.quaternion.rotate(la)
            )
            self.gizmo[i].position = wa

            # convert world anchor to B local space
            lb = self.bodyB.parent.quaternion.inverse().rotate(
                wa - self.bodyB.parent.position
            )

            self.localB.append(lb)
            self.rest_lengths.append(0.0)  # welded
        return "joint"
    # --------------------------------------------------
    # solve ONE distance constraint (this is your solver)
    # --------------------------------------------------
    def _solve_point(self, localA, localB, rest_length, dt):
        A = self.bodyA
        B = self.bodyB

        if A.isKinematic and B.isKinematic:
            return

        # world anchors
        rA = A.parent.quaternion.rotate(localA)
        rB = B.parent.quaternion.rotate(localB)

        xA = A.parent.position + rA
        xB = B.parent.position + rB

        d = xB - xA
        dist = d.magnitude()
        if dist == 0:
            return

        n = d / dist

        # velocities at anchor
        vA = A.velocity + A.angular_velocity.cross(rA)
        vB = B.velocity + B.angular_velocity.cross(rB)
        v_rel = vB - vA

        # constraint
        C = dist - rest_length
        beta = 0.2
        bias = beta * C / dt
        Cdot = v_rel.dot(n)

        # inverse mass
        invMassA = 0.0 if A.isKinematic else 1.0 / A.mass
        invMassB = 0.0 if B.isKinematic else 1.0 / B.mass

        invIA = A.Iinv_world()
        invIB = B.Iinv_world()

        raCn = rA.cross(n).to_np()
        rbCn = rB.cross(n).to_np()

        eff_mass = (
            invMassA + invMassB
            + raCn @ invIA @ raCn
            + rbCn @ invIB @ rbCn
        )

        if eff_mass == 0:
            return

        # impulse
        lambda_n = -(Cdot + bias) / eff_mass
        J = n * lambda_n

        # apply
        if not A.isKinematic:
            A.velocity -= J * invMassA
            A.angular_velocity -= Vector3.from_np(
                invIA @ rA.cross(J).to_np()
            )

        if not B.isKinematic:
            B.velocity += J * invMassB
            B.angular_velocity += Vector3.from_np(
                invIB @ rB.cross(J).to_np()
            )

    # --------------------------------------------------
    # solve FIXED joint (3 points)
    # --------------------------------------------------
    def solve(self, dt, iterations=10):
        for _ in range(iterations):
            for i in range(3):
                self._solve_point(
                    self.localA[i],
                    self.localB[i],
                    self.rest_lengths[i],
                    dt
                )
