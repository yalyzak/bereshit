import socket
import threading
import json
import random
import string
from concurrent.futures import ThreadPoolExecutor

from bereshit import World, Object, Vector3, Core

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

                # world = World()


                rooms[passcode] = {
                    "users": {},
                    # "world": world
                }

                # Core.run(world)

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

                # world = rooms[room]["world"]
                # world.add_object(Object(name=username))

                # OK: add user
                rooms[room]["users"][username] = (addr[0], udp_port)
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

def handle_packet(data, addr):
    try:
        msg = json.loads(data.decode())
    except Exception:
        return

    if msg.get("action") == "broadcast":
        username = msg["username"]
        message = msg["message"]
        room = user_rooms.get(username)
        if room:
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
