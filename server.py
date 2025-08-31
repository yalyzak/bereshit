import socket
import select
import time
from content_filter import ContentFilter
from user_manager import UserManager
from clientHandler import ClientHandler
class Server:
    def __init__(self,HOST = '127.0.0.1',PORT = 12345,user_manager=None,filter=None,hendler=None):
        self.HOST =HOST
        self.PORT = PORT
        self.attach(user_manager,filter,hendler)
    def attach(self,user_manager,filter,hendler):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.HOST, self.PORT))
        self.server_socket.listen()
        print(f"Server started on {self.HOST}:{self.PORT}")

        self.user_manager = user_manager if user_manager else UserManager()
        self.filter = filter if filter else ContentFilter()
        self.hendler = hendler if hendler else ClientHandler(self.server_socket)

        # Stores sockets flagged for delayed kick: {socket: timestamp_sent_message}
        self.users_to_kick = {}



    def hander_msg(self,header,message,ip,hendler,notified_socket):
        if header == 1:
            if self.filter.is_message_clean(message):
                hendler.clients[notified_socket]["username"] = message
                hendler.clients[notified_socket]["ip"] = ip
                hendler.broadcast(notified_socket, f"[SYSTEM] {message} username had joined")
            else:
                hendler.broadcast(notified_socket, f"[SYSTEM] {message} username is not Valid please pick another username")
                hendler.clients[notified_socket]["username"] = "john doe"


        elif header == 2:
            hendler.broadcast(notified_socket, f"[SYSTEM] {hendler.clients[notified_socket]['username']} has changed its name to {message}")
            hendler.clients[notified_socket]["username"] = message
        else:
            username = hendler.clients[notified_socket]["username"]
            if self.filter.is_message_clean(message):
                hendler.broadcast(notified_socket, f"{username}: {message}")
            else:
                ip = hendler.clients[notified_socket]["ip"]
                kick, violations = self.user_manager.report_offense(username, ip)
                if kick:
                    try:
                        notified_socket.send("[SYSTEM] You have been removed for inappropriate behavior.".encode())
                        self.users_to_kick[notified_socket] = time.time()
                        return notified_socket
                    except:
                        hendler.cleanup_client(notified_socket)
                else:
                    try:
                        notified_socket.send(
                            f"[WARNING] You have {violations} violations, extensive violations will cause ban.".encode())
                    except:
                        pass


