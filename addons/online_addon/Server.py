import socket
import struct
import threading
import json
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor

from bereshit import Object, Vector3, Core, Camera, Quaternion, BoxCollider, Rigidbody, Material
from bereshit.addons.essentials import debug
from bereshit.addons.online_addon import ServerController,InputMovementController

HOST = "0.0.0.0"
TCP_PORT = 5000
UDP_PORT = 5001


class User:
    def __init__(self, ip, port, name, room):
        self._ping = 0
        self._name = name
        self._room = room
        self.last_seen = 0
        self._ip = ip
        self._port = port

    def GetName(self):
        return self._name

    def Disconnect(self):
        self._room.RemovePlayer(self._name)
    def SetPing(self, ping):
        self._ping = ping
    def GetPing(self):
        return self._ping
    def GetIP(self):
        return self._ip

    def GetPort(self):
        return self._port


class Room:
    def __init__(self, world, password=None):
        self._PassWord = password if password else self.generate_passcode()
        self._players = {}
        self._World = world

    def generate_passcode(self, length=6):
        chars = string.ascii_uppercase + string.digits
        return ''.join(random.choices(chars, k=length))  # must return

    def AddPlayer(self, player):
        name = player.GetName()

        if name in self._players:
            return f"Player name '{name}' already exists in room."

        self._players[name] = player

    def RemovePlayer(self, name):
        if name in self._players:
            del self._players[name]

    def GetPassCode(self):
        return self._PassWord
    def GetWorld(self):
        return self._World

    def GetAllPlayers(self):
        return list(self._players.values())

    def GetPlayer(self, name):
        return self._players[name]

class RoomManager:
    def __init__(self):
        self._rooms = {}  # key: room_id, value: Room object

    def CreateRoom(self, world, password=None):
        room = Room(world, password)
        self._rooms[room._PassWord] = room
        return room.GetPassCode()
    def JointRoom(self, password, user):
        room = self.GetRoom(password)
        if room:
            return room.AddPlayer(user)
        else:
            return "Room not found"
    def RemoveRoom(self, password):
        if password in self._rooms:
            del self._rooms[password]

    def GetRoom(self, password):
        return self._rooms.get(password)

    def GetAllRooms(self):
        return self._rooms


manager = RoomManager()

def serverObject(name):
    return Object(name=name, position=Vector3(0, 4, 0)).add_component([BoxCollider(), Rigidbody(useGravity=False, Freeze_Rotation=Vector3(1,1,1)), InputMovementController()])

# ---------------------------------------------------
# TCP HANDLER (room creation, joining)
# ---------------------------------------------------

def handle_tcp_client(conn, addr):
    print("[TCP] Connection from", addr)

    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break

            msg = json.loads(data.decode())
            action = msg.get("action")


            # --- Create Room ---
            if action == "create_room":
                camera = Object(position=Vector3(0, 10, 0), rotation=Vector3(90, 0, 0)).add_component(Camera())

                room_id = manager.CreateRoom(camera)

                Map = [camera] + build_map()
                threading.Thread(target=Core.run, args=(Map,), kwargs={"Render": True}, daemon=False).start()

                conn.send(json.dumps({
                    "status": "ok",
                    "room": room_id
                }).encode())
                print("create_room")
            # --- Find Room ---
            elif action == "find_room":
                room = msg["room"]
                exists = manager.GetRoom(room) != None
                conn.send(json.dumps({"exists": exists}).encode())
                print("find_room")

            # --- Join Room ---
            elif action == "join_room":
                room = msg["room"]
                username = msg["username"]
                udp_port = msg["udp_port"]  # client tells us its UDP listening socket
                error = manager.JointRoom(room, User(addr[0],udp_port,username,room))
                if error:
                    conn.send(json.dumps({
                        "status": "error",
                        "message": error
                    }).encode())
                    # continue
                else:
                    world = manager.GetRoom(room).GetWorld()
                    world.add_child(serverObject(username))
                    conn.send(json.dumps({"status": "ok"}).encode())
                    print("join_room")


    except Exception as e:
        print("[ERROR]", e)

    finally:
        conn.close()
        print("[TCP] Closed", addr)


def tcp_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, TCP_PORT))
    s.listen()
    # print(f"[TCP] Listening on {HOST}:{TCP_PORT}")

    while True:
        conn, addr = s.accept()
        threading.Thread(target=handle_tcp_client, args=(conn, addr), daemon=True).start()


# ---------------------------------------------------
# UDP BROADCAST HANDLER
# ---------------------------------------------------

udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind((HOST, UDP_PORT))


def broadcast(room, sender, message):
    players = room.GetAllPlayers()
    for player in players:
        username, ip, port = player.GetName(), player.GetIP(), player.GetPort()
        payload = json.dumps({
            "from": sender,
            "message": message
        }).encode()
        udp_socket.sendto(payload, (ip, port))

    # for username, (ip, port) in rooms[room]["users"].items():
    #     if username != sender:
    #         payload = json.dumps({
    #             "room": room,
    #             "from": sender,
    #             "message": message
    #         }).encode()
    #
    #         udp_socket.sendto(payload, (ip, port))


executor = ThreadPoolExecutor(max_workers=8)


def build_map():
    objects = []

    def static_box(size, position, color):
        return (
            Object(size=size, position=position)
            .add_component([BoxCollider(), Rigidbody(isKinematic=True), Material(color=color)])
        )

    # --- Floor ---
    objects.append(
        static_box(
            size=Vector3(120, 1, 120),
            position=Vector3(0, -0.5, 0),
            color=(0.73, 0.75, 0.8)  # dark gray
        )
    )

    # --- Outer Walls (stone-like) ---
    wall_height = 12
    wall_thickness = 2
    arena_size = 60

    wall_color = (0.6, 0.6, 0.6)

    objects += [
        static_box(Vector3(arena_size * 2, wall_height, wall_thickness), Vector3(0, wall_height / 2, arena_size),
                   wall_color),
        static_box(Vector3(arena_size * 2, wall_height, wall_thickness), Vector3(0, wall_height / 2, -arena_size),
                   wall_color),
        static_box(Vector3(wall_thickness, wall_height, arena_size * 2), Vector3(arena_size, wall_height / 2, 0),
                   wall_color),
        static_box(Vector3(wall_thickness, wall_height, arena_size * 2), Vector3(-arena_size, wall_height / 2, 0),
                   wall_color),
    ]

    # --- Central Platform (sand color) ---
    objects.append(
        static_box(
            size=Vector3(20, 2, 20),
            position=Vector3(0, 1, 0),
            color=(1.0, 0.8, 0.4)
        )
    )

    # --- Elevated Side Platforms (green) ---
    platform_color = "green"
    objects += [
        static_box(Vector3(12, 2, 12), Vector3(25, 4, 25), platform_color),
        static_box(Vector3(12, 2, 12), Vector3(-25, 4, 25), platform_color),
        static_box(Vector3(12, 2, 12), Vector3(25, 4, -25), platform_color),
        static_box(Vector3(12, 2, 12), Vector3(-25, 4, -25), platform_color),
    ]

    # --- Ramps (orange) ---
    ramp_color = (1.0, 0.5, 0.2)
    objects += [
        static_box(Vector3(10, 2, 20), Vector3(0, 2, 20), ramp_color),
        static_box(Vector3(10, 2, 20), Vector3(0, 2, -20), ramp_color),
    ]

    # --- Cover / Obstacles (red) ---
    for x in [-30, -10, 10, 30]:
        objects.append(
            static_box(
                size=Vector3(4, 6, 4),
                position=Vector3(x, 3, 0),
                color="red"
            )
        )

    # --- Pillars (blue) ---
    for x, z in [(-20, -20), (20, -20), (-20, 20), (20, 20)]:
        objects.append(
            static_box(
                size=Vector3(3, 10, 3),
                position=Vector3(x, 5, z),
                color=(0.3, 0.5, 1.0)
            )
        )

    return objects


def move(player, data):
    # --- Ensure correct data size ---
    keys = data["keys"]

    if len(data) != 10:
        print("Bad message size from", len(data))
        return

        # Extract values
    px, py, pz = data[0:3]
    qx, qy, qz, qw = data[3:7]
    vx, vy, vz = data[7:10]

    player.position.x = px
    player.position.y = py
    player.position.z = pz

    player.quaternion = Quaternion(qx, qy, qz, qw)

    rb = player.get_component("Rigidbody")
    rb.velocity = Vector3(vx, vy, vz)


def handle_packet(data, addr):
    try:
        header = data[0]
        username = data[1:9].rstrip(b'\x00').decode()
        RoomCode = data[9:17].rstrip(b'\x00').decode()
        if header != 1:
            message = json.loads(data[17:].decode())



        # count = len(message) // 4
        # if count > 0:
        #     message = list(struct.unpack(f"!{count}f", message))
    except Exception:
        return
    room = manager.GetRoom(RoomCode)
    if room:
        user = room.GetPlayer(username)
        if user:
            if header == 0:
                world = room.GetWorld()
                obj = world.search_by_name(username)

                obj.InputMovementController.move_with_input(message)

                broadcast(room, username, message)
            elif header == 1:
                ip, _ = addr
                listen_port = user.GetPort()
                user.SetPing((time.perf_counter() - user.last_seen) * 1000)
                user.last_seen = time.perf_counter()
                udp_socket.sendto(b"HB", (ip, listen_port))
            # elif header == 2:
            #     world = room.GetWorld()
            #     obj = world.search_by_name(username)
            #     last_seen = user.last_seen
            #     obj.ServerController.server_controller(last_seen, message)
            #     broadcast(room, username, message)


def udp_server():
    print(f"[UDP] Listening on {HOST}:{UDP_PORT}")
    while True:
        try:
            data, addr = udp_socket.recvfrom(4096)
            executor.submit(handle_packet, data, addr)
        except ConnectionResetError:
            continue


def main():
    print("[SERVER] Starting...")

    threading.Thread(target=tcp_server, daemon=True).start()
    udp_server()  # UDP must stay on main thread


if __name__ == "__main__":
    main()

