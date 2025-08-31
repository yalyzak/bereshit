

class ClientHandler:
    def __init__(self,server_socket):
        self.clients = {}
        self.sockets_list = [server_socket]

    def broadcast(self,sender_socket, message):
        for client_socket in self.clients:
            # if client_socket != sender_socket:
                try:
                    client_socket.send(message.encode())
                except:
                    ClientHandler.cleanup_client(client_socket)

    def cleanup_client(self,client_socket):
        if client_socket in self.sockets_list:
            self.sockets_list.remove(client_socket)
        if client_socket in self.clients:
            del self.clients[client_socket]
        if client_socket in self.users_to_kick:
            del self.users_to_kick[client_socket]
        try:
            client_socket.close()
        except:
            pass

    def receive_message(self,sock):
        msg_type = sock.recv(1)
        if not msg_type:
            self.cleanup_client(sock)
            return None

        msg_type = msg_type[0]  # convert from bytes to int

        # 4 bytes message length (ascii)
        length_str = sock.recv(4).decode("ascii")
        length = int(length_str)

        # N bytes message body
        body = b""
        while len(body) < length:
            chunk = sock.recv(length - len(body))
            if not chunk:
                break
            body += chunk

        return msg_type, body.decode("utf-8")


