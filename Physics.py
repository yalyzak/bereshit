class RaycastHit:
    def __init__(self, point=None, normal=None, distance=None, collider=None, transform=None, rigidbody=None):
        self.point = point
        self.normal = normal
        self.distance = distance
        self.collider = collider
        self.transform = transform
        self.rigidbody = rigidbody

class Physics:
    World = None
    def __init__(self, origin, direction, maxDistance=float('inf'), hit=None):
        self.origin = origin
        self.direction = direction
        self.maxDistance = maxDistance

    @staticmethod
    def Raycast(origin, direction, layerMask=None, maxDistance=float('inf')):
        origin = origin.to_np()
        direction = direction.to_np()

        hit = RaycastHit()
        if layerMask:
            hit = layerMask.Raycast(origin, direction, maxDistance)
        else:
            dis = float('inf')
            for object in Physics.World.get_all_children():
                collider = object.get_component("collider")
                if collider:
                    temphit = collider.Raycast(origin, direction, maxDistance)
                    if temphit.point is not None and temphit.distance < dis:
                        dis = temphit.distance
                        hit = temphit
        return hit

    @staticmethod
    def RaycastAll(origin, direction, maxDistance=float('inf')):
        origin = origin.to_np()
        direction = direction.to_np()
        hits = []
        for object in Physics.World.get_all_children():
            collider = object.get_component("Collider")
            if collider:
                hit = collider.Raycast(origin, direction, maxDistance)
                if hit.point is not None:
                    hits.append(hit)

        return hits

