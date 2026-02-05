import traceback
import time

import numpy as np

from bereshit.Quaternion import Quaternion
from bereshit.Rigidbody import Rigidbody
from bereshit.Vector3 import Vector3

class World:
    def __init__(self, children=None,gizmos=None,gravity=Vector3(0, -9.8, 0),tick=None,speed=None):
        self.children = children or []
        self.Camera = self.search_by_component('Camera')
        self.gizmos = gizmos
        World.Gravity = gravity
        World.tick = tick
        World.speed = speed
        World.Objects = self.get_all_children()



    def add_object(self, object):
        self.children.append(object)
    def search_by_component(self, component_name):
        # Check if this object has the desired component
        if hasattr(self, "components") and component_name in self.components:
            return self
        # Recursively check children
        if hasattr(self, "children"):
            for child in self.children:
                result = child.search_by_component(component_name)
                if result:
                    return result
        return None

    def search_by_name(self, object_name):
        for child in self.children:
            result = child.search_by_name(object_name)
            if result:
                return result
        return None


    def get_all_children(self):
        all_objs = []
        for child in self.children:
            target = child
            all_objs.append(target)
            all_objs.extend(target.get_all_children())
        return all_objs
    def get_all_children_physics(self):
        all_objs = []
        for child in self.children:
            rb = child.get_component("Rigidbody")
            collider = child.get_component("collider")
            if rb and collider:
                all_objs.append(child)
            all_objs.extend(child.get_all_children_physics())
        return all_objs

    def apply_gravity(self,children):
        for child in children:
            rb = child.get_component("Rigidbody")
            rb.apply_gravity()

    def solve_collections(self, children, dt, gizmos):

        contacts = []
        contacts2 = []
        beta = 0.0  # softness factor for positional correction

        # STEP 1: Collect all contacts (use ALL manifold points)
        for i in range(len(children)):
            obj1 = children[i]
            rb1 = obj1.get_component("Rigidbody")

            for j in range(i + 1, len(children)):
                obj2 = children[j]
                rb2 = obj2.get_component("Rigidbody")

                # Skip if neither has a Rigidbody or both are kinematic
                if (rb1 is None or rb1.isKinematic) and (rb2 is None or rb2.isKinematic):
                    continue

                result = obj1.collider.check_collision(obj2, single_point=True)
                if result is None:
                    continue

                contact_points, rb = result  # contact_points = [(cp, n, pn), ...]
                c1, c2 = rb
                rb1, rb2 = c1.parent.Rigidbody, c2.parent.Rigidbody

                if type(contact_points[0]) == tuple:  # For each point in the manifold, add a separate constraint
                    for (contact_point, normal, penetration) in contact_points:
                        contact_point = Vector3(contact_point)
                        r1 = contact_point - rb1.parent.position
                        r2 = contact_point - rb2.parent.position

                        v1_linear = rb1.velocity if (rb1 and not rb1.isKinematic) else Vector3(0, 0, 0)
                        v2_linear = rb2.velocity if (rb2 and not rb2.isKinematic) else Vector3(0, 0, 0)
                        v1_angular = rb1.angular_velocity.cross(r1) if (rb1 and not rb1.isKinematic) else Vector3(0, 0,
                                                                                                                  0)
                        v2_angular = rb2.angular_velocity.cross(r2) if (rb2 and not rb2.isKinematic) else Vector3(0, 0,
                                                                                                                  0)

                        v_rel_linear = v1_linear - v2_linear  # B minus A (matches normal pointing A->B)
                        v_norm_linear = v_rel_linear.dot(normal)

                        v_rel_angular = v1_angular - v2_angular  # B minus A (matches normal pointing A->B)
                        v_norm_angular = v_rel_angular.dot(normal)

                        v_rel = (v1_linear + v1_angular) - (v2_linear + v2_angular)
                        v_norm = v_rel.dot(normal)

                        contacts2.append({
                            "j1" : 0,
                            "r1": r1,
                            "r2": r2,
                            "rb1": rb1,
                            "rb2": rb2,
                            "normal": normal,
                            "v_norm": v_norm,
                            "v_norm_linear": v_norm_linear,
                            "v_norm_angular": v_norm_angular,
                            "penetration": penetration,
                            "contact_point": contact_point,
                        })
                    contacts.append(contacts2)

                elif type(contact_points[0]) == Vector3:
                    contact_point, normal, penetration = contact_points

                    r1 = contact_point - rb1.parent.position
                    r2 = contact_point - rb2.parent.position

                    if not rb1.isKinematic:
                        rb1.velocity += (rb1.force / rb1.mass) * dt
                        rb1.force = Vector3()
                    if not rb2.isKinematic:
                        rb2.velocity += (rb2.force / rb2.mass) * dt
                        rb2.force = Vector3()
                    v1_linear = rb1.velocity if (rb1 and not rb1.isKinematic) else Vector3(0, 0, 0)
                    v2_linear = rb2.velocity if (rb2 and not rb2.isKinematic) else Vector3(0, 0, 0)
                    v1_angular = rb1.angular_velocity.cross(r1) if (rb1 and not rb1.isKinematic) else Vector3(0, 0, 0)
                    v2_angular = rb2.angular_velocity.cross(r2) if (rb2 and not rb2.isKinematic) else Vector3(0, 0, 0)

                    v_rel_linear = v1_linear - v2_linear  # B minus A (matches normal pointing A->B)
                    v_norm_linear = v_rel_linear.dot(normal)

                    v_rel_angular = v1_angular - v2_angular  # B minus A (matches normal pointing A->B)
                    v_norm_angular = v_rel_angular.dot(normal)

                    v_rel = (v1_linear + v1_angular) - (v2_linear + v2_angular)
                    v_norm = v_rel.dot(normal)
                    contacts.append([{
                        "j1": 0,
                        "r1": r1,
                        "r2": r2,
                        "rb1": rb1,
                        "rb2": rb2,
                        "normal": normal,
                        "v_norm": v_norm,
                        "v_norm_linear": v_norm_linear,
                        "v_norm_angular": v_norm_angular,
                        "penetration": penetration,
                        "contact_point": contact_point,
                    }])

        if gizmos:
            self.set_gizmos(contacts=contacts)
        N = 0
        for i in range(len(contacts)):
            N += len(contacts[i])
        if N == 0:
            return contacts

        k = np.zeros((N, 2))
        for contact_point in contacts:
            length = len(contact_point)
            for i, c in enumerate(contact_point):
                rb1 = c["rb1"]
                rb2 = c["rb2"]
                restitution = 0.0
                if c["v_norm"] > -1:
                    restitution
                elif rb1 and rb2:
                    restitution = min(rb1.restitution, rb2.restitution)
                elif rb1:
                    restitution = rb1.restitution
                elif rb2:
                    restitution = rb2.restitution

                rn1 = c["r1"].cross(c["normal"])
                rn2 = c["r2"].cross(c["normal"])

                if not rb1.isKinematic:
                    term1 = (Vector3.from_np(Iinv_world(rb1) @ rn1.to_np())).cross(c["r1"])
                else:
                    term1 = Vector3(0, 0, 0)

                if not rb2.isKinematic:
                    term2 = (Vector3.from_np(Iinv_world(rb2) @ rn2.to_np())).cross(c["r2"])
                else:
                    term2 = Vector3(0, 0, 0)

                v1_at_p = rb1.velocity + rb1.angular_velocity.cross(c["r1"])
                v2_at_p = rb2.velocity + rb2.angular_velocity.cross(c["r2"])
                relative_vel = v2_at_p - v1_at_p
                v_norm = relative_vel.dot(c["normal"])

                k_linear = (0 if rb1.isKinematic else 1 / rb1.mass) + (0 if rb2.isKinematic else 1 / rb2.mass)
                k_angular = c["normal"].dot(term1 + term2)
                inv_eff_mass = k_linear + k_angular

                c["J1"] = (1 + restitution) * v_norm / inv_eff_mass



        for contact_point in contacts:
            for i, contact in enumerate(contact_point):
                if contact["v_norm"] >= 0.1:
                    continue
                normal = contact["normal"]
                contact_point = contact["contact_point"]
                r1 = contact["r1"]
                r2 = contact["r2"]
                J1 = contact["J1"]
                rb1 = contact["rb1"]
                rb2 = contact["rb2"]

                rb1.resolve_dynamic_collision(rb2, normal, J1, r1, r2)
                rb1.apply_friction_impulse(rb2, normal, J1, contact_point)

        return contacts

    def apply_friction(self, contact, normal):
        rb1 = contact["rb1"]
        rb2 = contact["rb2"]
        contact_point = contact["contact_point"]  # required for torque calculation
        v1 = rb1.velocity if rb1 and not rb1.isKinematic else Vector3(0, 0, 0)
        v2 = rb2.velocity if rb2 and not rb2.isKinematic else Vector3(0, 0, 0)
        v_rel = v1 - v2
        f1 = rb1.force if rb1 and not rb1.isKinematic else Vector3(0, 0, 0)
        f2 = rb2.force if rb2 and not rb2.isKinematic else Vector3(0, 0, 0)
        f_rel = f1 - f2

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
        Jt_magnitude = -f_rel.dot(tangent)

        denom = 0.0
        if rb1 and not rb1.isKinematic:
            denom += 1.0 / rb1.mass
        if rb2 and not rb2.isKinematic:
            denom += 1.0 / rb2.mass

        if denom == 0.0:
            return

    def apply_friction_impulse(self, contact, normal, Jn):
        """
        Applies Coulomb friction impulse based on the relative velocity,
        including angular friction effect.
        """
        rb1 = contact["rb1"]
        rb2 = contact["rb2"]
        contact_point = contact["contact_point"]  # required for torque calculation

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
        max_friction = mu * Jn
        Jt_magnitude = max(-max_friction, min(Jt_magnitude, max_friction))

        Jt = tangent * Jt_magnitude

        # Apply linear and angular friction impulses
        if rb1 and not rb1.isKinematic:
            rb1.velocity += Jt / rb1.mass
            r1 = contact_point - rb1.parent.position
            angular_impulse1 = r1.cross(Jt)
            # rb1.angular_velocity += Vector3.from_np(rb1.inverse_inertia @ angular_impulse1.to_np())

        if rb2 and not rb2.isKinematic:
            rb2.velocity -= Jt / rb2.mass
            r2 = contact["r2"]
            angular_impulse2 = r2.cross(-Jt)
            # rb2.angular_velocity += Vector3.from_np(rb2.inverse_inertia @ angular_impulse2.to_np())
    def position_correctness(self,contact,point,depth):
        n = contact["normal"]
        vector = n * depth
        contact["rb2"].velocity -= vector
    def resolve_dynamic_collision(self, contact, J, flage2, flage):
        """
        Applies linear and angular impulse to both dynamic bodies, factoring restitution.
        """
        n = contact["normal"]
        contact_point = contact["contact_point"]  # world-space contact point
        rb1 = contact["rb1"]
        rb2 = contact["rb2"]

        impulse_vec = n * J
        # impulse_vec = n * J2

        if rb1 and not rb1.isKinematic:
            if flage:
                rb1.force += -rb1.force * rb1.mass * n
                rb1.velocity += rb1.velocity * n
            if flage2:
                rb1.angular_velocity = Vector3()

            rb1.velocity += impulse_vec / rb1.mass
            torque_impulse = contact["r1"].cross(impulse_vec)
            ang_impulse = Vector3.from_np(Iinv_world(rb1) @ torque_impulse.to_np())
            rb1.angular_velocity -= ang_impulse
            # else:
            #     torque_impulse = contact["r1"].cross(impulse_vec)
            #     ang_impulse = Vector3.from_np(Iinv_world(rb1) @ torque_impulse.to_np())
            #     rb1.angular_velocity += ang_impulse
        if rb2 and not rb2.isKinematic:
            if flage:
                # pass
                rb2.force += -rb1.force * rb2.mass * n
                rb2.velocity += rb2.velocity * n
            if flage2:

                rb2.angular_velocity = Vector3()

            # if not flage2:
            rb2.velocity -= impulse_vec / rb2.mass
            torque_impulse = contact["r2"].cross(impulse_vec)
            ang_impulse = Vector3.from_np(Iinv_world(rb2) @ torque_impulse.to_np())
            rb2.angular_velocity += ang_impulse
            # else:
            #     torque_impulse = contact["r2"].cross(impulse_vec)
            #     ang_impulse = Vector3.from_np(Iinv_world(rb2) @ torque_impulse.to_np())
            #     rb2.angular_velocity -= ang_impulse
            # else:
            #     rb2.velocity -= impulse_vec / rb2.mass


    def resolve_kinematic_collision(self, contact, J, J2, flage):
        """
        Applies linear and angular impulse to the dynamic body only, factoring restitution.
        """
        n = contact["normal"]
        contact_point = contact["contact_point"]  # world-space contact point
        rb1 = contact["rb1"]
        rb2 = contact["rb2"]

        impulse_vec = n * J
        impulse_vec2 = n * J2

        if rb1 and not rb1.isKinematic:
            if flage:
                rb1.force += -rb1.force * rb1.mass * n
                rb1.velocity += rb1.velocity * -n
                return

            # 1. Linear Change: dv = J / m
            rb1.velocity += impulse_vec / rb1.mass

            # 2. Angular Change: dw = I^-1 * (r x J)
            torque_impulse1 = contact["r1"].cross(impulse_vec)
            # Use the inverse world inertia tensor (3x3 matrix @ vector)

            # rb1.angular_velocity += Vector3.from_np(Iinv_world(rb1) @ torque_impulse1.to_np())

        if rb2 and not rb2.isKinematic:
            if flage:
                rb2.force += rb2.force * rb2.mass * n
                rb2.velocity += rb2.velocity * n
                return


            # 1. Linear Change (Opposite direction)
            rb2.velocity -= impulse_vec / rb2.mass

            # 2. Angular Change (r2 x -impulse)
            torque_impulse2 = contact["r1"].cross(impulse_vec)
            # rb2.angular_velocity += Vector3.from_np(Iinv_world(rb2) @ torque_impulse2.to_np())

    def set_gizmos(self, contacts=[]):
        g = False
        for i, contact_point in enumerate(contacts):
            self.gizmos.children[i].position = contact_point[0]['contact_point']


                    # self.children[1].children[i].children[j].position = contact['ref_face_center'][j]
                    #
                    # self.children[1].children[i].children[j + 4].position = contact['incident_face'][j]
        # while g:
        #     pass
        # Object(position=contacts['contact_point'])

    @staticmethod
    def solve_joints(children, dt):
        """
        Go through all objects, find joints, and solve their constraints.
        """
        for child in children:
            joint = child.get_component("joint")
            if joint is not None:
                joint.solve(dt)

    def Start(self):
        children1 = self.get_all_children()
        for child in children1:
            for component in child.components.values():
                if hasattr(component, 'Start') and component.Start is not None and component.Active == True:
                    try:
                        component.Start()
                    except:
                        print(f"[Error] Exception in {component.__class__.__name__}.Start():")
                        traceback.print_exc()

    def update(self, check=True, gizmos=False):
        dt = World.tick
        allchildren = self.get_all_children()
        if check:
            for child in allchildren:
                for component in child.components.values():
                    if hasattr(component, 'Update') and component.Update is not None and component.Active == True:
                        try:
                            component.Update(dt)
                        except Exception as e:
                            print(f"[Error] Exception in {component.__class__.__name__}.Update(): {e}")
                            traceback.print_exc()

        children = self.get_all_children_physics()
        self.apply_gravity(children)  # APPLY GRAVITY and external forces
        for _ in range(1):
            self.solve_collections(children, dt, gizmos)  # handel collisions and friction
        self.solve_joints(children, dt)

        for child in children:
            rb = child.get_component("Rigidbody")
            if rb is not None:
                child_children = child.get_all_children_not_physics()
                for child_of_child in child_children:
                    if child_of_child.get_component("Rigidbody") is not None:
                        child_of_child.position += child_of_child.Rigidbody.velocity * dt \
                                                   + 0.5 * child_of_child.Rigidbody.acceleration * dt * dt
                if not rb.isKinematic:
                    child.Rigidbody.integrat(dt)

        for child in allchildren:
            child.rotation = child.quaternion.to_euler()
def Iinv_world(rb):
    # This should only be called for dynamic bodies
    if not rb or rb.isKinematic:
        return np.zeros((3, 3))

    # 1. Get the local inverse inertia tensor (3x3 matrix)
    # Ensure this is the INVERSE, not the base inertia
    I_inv_body = rb.inverse_inertia

    # 2. Get the rotation matrix
    # If using a quaternion:
    R = rb.parent.quaternion.to_matrix3()
    # R = Quaternion().to_matrix3()

    # 3. Transform to world space: R * I_inv * R_transpose
    return R @ I_inv_body @ R.T
# def Iinv_world(rb):
#     R = rb.parent.quaternion.to_matrix3()  # 3x3 from quaternion
#
#     I_local = np.diag([rb.inertia.x, rb.inertia.y, rb.inertia.z])
#     I_world = R @ I_local @ R.T
#     I_world_inv = np.linalg.inv(I_world)
#     return Vector3.from_np(I_world_inv)