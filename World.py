import random
import traceback
import time
import logging
import numpy as np

from bereshit.Quaternion import Quaternion
from bereshit.Rigidbody import Rigidbody
from bereshit.Vector3 import Vector3
from bereshit.ContactPoint import ContactPoint
from bereshit.ContactPoint import ContactManifold
from bereshit.class_type import Joint
from bereshit.Collider import Collider

logger = logging.getLogger(__name__)
class World:
    def __init__(self, running_flag, children=None, gizmos=False, gravity=Vector3(0, -9.8, 0), tick=None, speed=None, physics_epochs=1):
        self.RunningFlag = running_flag
        self.children = children or []
        self.Camera = self.search_by_component('Camera')
        self.gizmos = gizmos
        self.manifold_cache = {}
        self.gravity = gravity
        self.tick = tick
        self.speed = speed
        self.objects = self.get_all_children()
        for obj in self.objects:
            obj.World = self

        self.physics_epochs = physics_epochs
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
            rb = child.get_component(Rigidbody)
            collider = child.get_component(Collider)
            if rb and not collider:
                logger.warning(f"object {child.name} has a Rigidbody but no Collider")
            if collider and not rb:
                logger.warning(f"object {child.name} has a Rigidbody but no Collider")
            if rb and collider:
                all_objs.append(child)
            all_objs.extend(child.get_all_children_physics())
        return all_objs

    def apply_gravity(self, children):
        for child in children:
            rb = child.get_component("Rigidbody")
            rb.apply_gravity(self.gravity)

    @staticmethod
    def solve_joints(children, dt):
        """
        Go through all objects, find joints, and solve their constraints.
        """
        for child in children:
            joints = child.get_all_components(Joint)
            for joint in joints:
                joint.solve(dt)

    def solve_collections(self, dt, contacts):
        for contact in contacts:
            contact_point = contact['contact_point']
            normal = contact['normal']
            rb1, rb2 = contact['rb1'], contact['rb2']
            penetration = contact['penetration']
            Rigidbody.solve_impulse(rb1, rb2, contact_point, normal, penetration, dt, apply_friction=True)

    def solve_collectionsFirstIteration(self, children, dt):

        contacts = []
        # STEP 1: Collect all contacts (use ALL manifold points)
        for i in range(len(children)):
            first_obj = children[i]
            first_rb = first_obj.get_component("Rigidbody")
            first_collider = first_obj.get_component("Collider")
            collided_a = False
            for j in range(i + 1, len(children)):
                second_obj = children[j]
                second_rb = second_obj.get_component("Rigidbody")
                second_collider = second_obj.get_component("Collider")
                collided_b = False

                # Skip if neither has a Rigidbody or both are kinematic
                if (first_rb is None or first_rb.isKinematic) and (second_rb is None or second_rb.isKinematic):
                    continue

                result = first_collider.check_collision(second_collider, single_point=False, collided_a=collided_a,
                                                        collided_b=collided_b)
                if result is None:
                    continue
                contacts2 = []
                collided_a, collided_b = True, True
                rb1, rb2 = first_rb, second_rb


                for contact_point in result.contact_points:
                    Rigidbody.solve_impulse(rb1, rb2, contact_point, result.normal, result.depth, dt, apply_friction=True)

                    contacts.append({
                        "rb1": rb1,
                        "rb2": rb2,
                        "normal": result.normal,
                        "penetration": result.depth,
                        "contact_point": contact_point,
                    })


        if self.gizmos:
            self.set_gizmos(contacts=contacts)  # needs Updating/Fixing

        return contacts

    def set_gizmos(self, contacts=[]):
        for i, contact_point in enumerate(contacts):
            self.gizmos.children[i].position = contact_point['contact_point']

    def set_gizmos2(self, contacts=[]):
        for i, contact_point in enumerate(contacts):
            self.gizmos.children[i].position = contact_point

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
                        exit()

    def update(self, check=True, gizmos=False):
        dt = self.tick
        FirstIteration = True
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
        contacts = self.solve_collectionsFirstIteration(children, dt)  # handel collisions and friction
        self.solve_joints(children, dt)

        for _ in range(self.physics_epochs - 1):  # keeps Constraint inline
            self.solve_collections(dt, contacts)  # handel collisions and friction
            self.solve_joints(children, dt)

        # self.solve_collectionsLastIteration(dt, contacts)

        for child in children:
            rb = child.get_component("Rigidbody")
            if rb is not None:
                child_children = child.get_all_children_not_physics()
                for child_of_child in child_children:
                    if child_of_child.get_component("Rigidbody") is not None:
                        child_of_child.position += child_of_child.Rigidbody.velocity * dt \
                                                   + 0.5 * child_of_child.Rigidbody.acceleration * dt * dt
                    if not child_of_child.get_component("Rigidbody"):
                        child_of_child.position = child.position + child_of_child.default_position
                        child_of_child.quaternion = Quaternion.euler(child.rotation + child_of_child.default_rotation)

                if not rb.isKinematic:
                    child.Rigidbody.integrat(dt)

        for child in allchildren:
            child.rotation = child.quaternion.to_euler()

    def Exit(self, code=None):
        self.RunningFlag[0] = True

    def _remove_object(self, child):
        if child in self.children:
            self.children.remove(child)
