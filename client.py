import socket
import threading
import queue
from ui import ChatUI

class Client:
    def __init__(self, host='127.0.0.1', port=12345):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.message_queue = queue.Queue()

    def send(self, message,msg_type):
        body_bytes = message.encode("utf-8")
        length_str = f"{len(body_bytes):04d}"  # 4 ASCII digits
        packet = bytes([msg_type]) + length_str.encode("ascii") + body_bytes
        self.sock.sendall(packet)

    def receive_loop(self):
        while True:
            try:
                message = self.sock.recv(1024).decode()
                self.message_queue.put(message)
            except:
                break

    def run(self):
        ui = ChatUI(self)
        # username = None
        # while not username:
        #     username = input("Enter username: ").strip()
        # self.send(username)

        threading.Thread(target=self.receive_loop, daemon=True).start()
        ui.start()

if __name__ == "__main__":
    client = Client()
    client.run()
