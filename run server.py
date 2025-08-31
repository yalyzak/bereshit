import socket
import select
import time
from content_filter import ContentFilter
from user_manager import UserManager
from clientHandler import ClientHandler
from server import Server
content_filter = ContentFilter(UseGenai=True,UseNLP=False)
server = Server(filter=content_filter)

def run(server):
    while True:
        read_sockets, _, _ = select.select(server.hendler.sockets_list, [], [], 0.1)  # non-blocking with timeout

        for notified_socket in read_sockets:
            if notified_socket == server.server_socket:
                client_socket, client_address = server.server_socket.accept()
                ip = client_address[0]
                if server.user_manager.is_banned(ip):
                    print(f"Rejected banned IP: {ip}")
                    try:
                        client_socket.send("[SYSTEM] You are banned from this server.".encode())
                    except:
                        pass
                server.hendler.sockets_list.append(client_socket)
                server.hendler.clients[client_socket] = {"username": None}
                print(f"New connection from {client_address}")
            else:
                try:
                    # message = notified_socket.recv(1024).decode().strip()
                    header,message = server.hendler.receive_message(notified_socket)
                    server.hander_msg(header,message,ip,server.hendler,notified_socket)

                except:
                    server.hendler.cleanup_client(notified_socket)

        # Handle delayed kicks
        now = time.time()
        for sock in list(server.users_to_kick.keys()):
            if now - server.users_to_kick[sock] > 0.5:  # 0.5 seconds delay
                ClientHandler.cleanup_client(sock)
if __name__ == "__main__":
    run(server)