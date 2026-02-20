import socket
import time
from collections import deque
from bereshit import World, Object, Rigidbody, BoxCollider, Vector3, Quaternion, MeshRander
from scene.Goal import Goal
from bereshit.addons.essentials import PlayerController



players = {}  # { "username": Object() }

class Clientuse:
    def __init__(self, data_objects=[]):
        self.data_objects = data_objects
        self.UserName = None
        self.RoomName = None

        self.incoming = deque()  # stores raw bytes
        self.outgoing = deque()  # stores raw bytes
        self.ping_sent = False
        self.ping = 0
    def Start(self):
        self.Broadcast = self.parent.Client.Broadcast
        self.ReceiveMassages = self.parent.Client.ReceiveMassages

        self.player = Object()
        self.player.add_component(BoxCollider())
        self.player.add_component(Rigidbody(useGravity=True))
        ServerContiner = Object()
        ServerContiner.add_component(MeshRander(shape="empty"))
        self.parent.add_child(ServerContiner)
        self.Continer = ServerContiner
        self.inputs = self.parent.get_component("PlayerController").get_next_input

    def Send(self):
        for obj in self.data_objects:
            msg = [
                obj.position.x, obj.position.y, obj.position.z,
                obj.quaternion.x, obj.quaternion.y, obj.quaternion.z, obj.quaternion.w,

                obj.Rigidbody.velocity.x, obj.Rigidbody.velocity.y, obj.Rigidbody.velocity.z,

            ]
            self.Broadcast(b'\x00',self.UserName, self.RoomName, msg)

    def Send2(self):
        msg = self.inputs()
        self.Broadcast(b'\x00', self.UserName, self.RoomName, msg)
    def Update(self, dt):
        if self.UserName is not None:
            self.Send2()
            if not self.ping_sent:
                self.ping = time.perf_counter()
                self.Broadcast(b'\x01', self.UserName, self.RoomName, b"HB")

        msg = self.ReceiveMassages()
        if not msg:
            return
        if msg == "HB":
            rtt = (time.perf_counter() - self.ping) * 1000
            self.ping = rtt
            # print(f"Ping: {rtt:.2f} ms")
            self.ping_sent = False
        # else:



