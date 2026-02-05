import socket
import struct
import threading
import json
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor

from bereshit import Object, Vector3, Core, Camera, Quaternion, BoxCollider, Rigidbody, Material
from bereshit.addons.online_addon import ServerController

HOST = "0.0.0.0"
TCP_PORT = 5000
UDP_PORT = 5001

rooms = {}
user_rooms = {}
# rooms[passcode] = {
#    "users": {
#        username: (ip, udp_port)
#    }
# }

# ---------------------------------------------------
# Utility
# ---------------------------------------------------
class room:
    def __init__(self):
        self.passcode = None
        self.world = None
class User:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.ping = 0
        self.last_seen = 0

def generate_passcode(length=6):
    chars = string.ascii_uppercase + string.digits
    return "0"
    return ''.join(random.choices(chars, k=length))


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
                passcode = generate_passcode()

                empty = Object(position=Vector3(0,10,0),rotation=Vector3(90,0,0)).add_component(Camera())


                rooms[passcode] = {
                    "users": {},
                    "world": empty
                }

                Map = [empty]+build_map()
                threading.Thread(target=Core.run, args=(Map,), kwargs={"Render": True}, daemon=False).start()

                conn.send(json.dumps({
                    "status": "ok",
                    "room": passcode
                }).encode())
                print("create_room")
            # --- Find Room ---
            elif action == "find_room":
                room = msg["room"]
                exists = room in rooms
                conn.send(json.dumps({"exists": exists}).encode())
                print("find_room")

            # --- Join Room ---
            elif action == "join_room":
                room = msg["room"]
                username = msg["username"]
                udp_port = msg["udp_port"]  # client tells us its UDP listening socket

                # Room not found
                if room not in rooms:
                    conn.send(json.dumps({
                        "status": "error",
                        "message": "Room not found"
                    }).encode())
                    print("room not found")

                    continue

                # Username already exists
                if username in rooms[room]["users"]:
                    conn.send(json.dumps({
                        "status": "error",
                        "message": "Username already taken"
                    }).encode())
                    print("Username already exists")

                    continue

                world = rooms[room]["world"]
                world.add_child(Object(name=username, position=Vector3(0,5,0)).add_component([BoxCollider(),Rigidbody(useGravity=True), ServerController()]))

                # OK: add user
                rooms[room]["users"][username] = User(addr[0], udp_port)
                user_rooms[username] = room
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
    if room not in rooms:
        return
    for username, (ip, port) in rooms[room]["users"].items():
        if username != sender:
            payload = json.dumps({
                "room": room,
                "from": sender,
                "message": message
            }).encode()

            udp_socket.sendto(payload, (ip, port))

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
        static_box(Vector3(arena_size*2, wall_height, wall_thickness), Vector3(0, wall_height/2,  arena_size), wall_color),
        static_box(Vector3(arena_size*2, wall_height, wall_thickness), Vector3(0, wall_height/2, -arena_size), wall_color),
        static_box(Vector3(wall_thickness, wall_height, arena_size*2), Vector3(arena_size, wall_height/2, 0), wall_color),
        static_box(Vector3(wall_thickness, wall_height, arena_size*2), Vector3(-arena_size, wall_height/2, 0), wall_color),
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

        payload = data[9:]
        count = len(payload) // 4
        if count > 0:
            message = list(struct.unpack(f"!{count}f", payload))
    except Exception:
        return
    room = user_rooms.get(username)
    if room:
        if header == 0:
            world = rooms[room]["world"]
            obj = world.search_by_name(username)
            move(obj, message)
            broadcast(room, username, message)
        elif header == 1:
            ip, _ = addr
            user = rooms[room]["users"][username]
            listen_port = user.port
            user.ping = (time.perf_counter() - user.last_seen) * 1000
            user.last_seen = time.perf_counter()
            udp_socket.sendto(b"HB", (ip, listen_port))
        elif header == 2:
            obj = rooms[room]["world"].search_by_name(username)
            last_seen = rooms[room]["users"][username].last_seen
            obj.ServerController.server_controller(last_seen, message)
            broadcast(room, username, message)



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
    udp_server()   # UDP must stay on main thread


if __name__ == "__main__":
    main()
