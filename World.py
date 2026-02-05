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
    @staticmethod
    def apply_gravity(children):
        for child in children:
            rb = child.get_component("Rigidbody")
            rb.apply_gravity()
    @staticmethod
    def solve_joints(children, dt):
        """
        Go through all objects, find joints, and solve their constraints.
        """
        for child in children:
            joint = child.get_component("joint")
            if joint is not None:
                joint.solve(dt)
    def solve_collections(self, children, dt, gizmos):

        contacts = []
        contacts2 = []

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

        for contact_point in contacts:
            for i, contact in enumerate(contact_point):
                v_norm = contact["v_norm"]
                if v_norm >= 0.1:
                    continue

                normal = contact["normal"]
                contact_point = contact["contact_point"]
                r1 = contact["r1"]
                r2 = contact["r2"]
                rb1 = contact["rb1"]
                rb2 = contact["rb2"]
                restitution = rb1.find_restitution(rb2, v_norm)
                rn1 = r1.cross(normal)
                rn2 = r2.cross(normal)

                if not rb1.isKinematic:
                    term1 = (Vector3.from_np(rb1.Iinv_world() @ rn1.to_np())).cross(r1)
                else:
                    term1 = Vector3(0, 0, 0)

                if not rb2.isKinematic:
                    term2 = (Vector3.from_np(rb2.Iinv_world() @ rn2.to_np())).cross(r2)
                else:
                    term2 = Vector3(0, 0, 0)

                v1_at_p = rb1.velocity + rb1.angular_velocity.cross(r1)
                v2_at_p = rb2.velocity + rb2.angular_velocity.cross(r2)
                relative_vel = v2_at_p - v1_at_p
                v_norm = relative_vel.dot(normal)

                k_linear = (0 if rb1.isKinematic else 1 / rb1.mass) + (0 if rb2.isKinematic else 1 / rb2.mass)
                k_angular = normal.dot(term1 + term2)
                inv_eff_mass = k_linear + k_angular

                J = (1 + restitution) * v_norm / inv_eff_mass

                rb1.resolve_dynamic_collision(rb2, normal, J, r1, r2)
                rb1.apply_friction_impulse(rb2, normal, J, contact_point)

        return contacts

    def set_gizmos(self, contacts=[]):
        for i, contact_point in enumerate(contacts):
            self.gizmos.children[i].position = contact_point[0]['contact_point']

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
