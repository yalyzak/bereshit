import selectors
import socket

sel = selectors.DefaultSelector()

def accept(sock):
    conn, addr = sock.accept()
    print("Accepted connection from", addr)
    conn.setblocking(False)
    sel.register(conn, selectors.EVENT_READ, read)

def read(conn):
    try:
        data = conn.recv(1024)
        if data:
            print("Received:", data.decode())
            conn.sendall(data)
        else:
            print("Closing connection")
            sel.unregister(conn)
            conn.close()
    except ConnectionResetError:
        print("Client reset connection")
        sel.unregister(conn)
        conn.close()


# Create listening socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("0.0.0.0", 5000))
sock.listen()
sock.setblocking(False)

# Register the listening socket for READ events
sel.register(sock, selectors.EVENT_READ, accept)

print("Server listening on port 5000...")

# Event loop
while True:
    events = sel.select(timeout=None)  # Blocks until something is ready
    for key, mask in events:
        callback = key.data  # The function we attached (accept/read)
        callback(key.fileobj)
