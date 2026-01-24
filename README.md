    def Broadcast(self, header, UserName, message):
        """Send message to server's UDP broadcast system (binary protocol)."""

        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # --- header: 1 byte ---
        if isinstance(header, str):
            header = header.encode()
        header_byte = header[:1]  # ensure exactly 1 byte

        # --- username: 8 bytes ---
        if isinstance(UserName, str):
            UserName = UserName.encode("utf-8")

        username_bytes = UserName[:8].ljust(8, b'\x00')  # pad or cut to 8 bytes
        if isinstance(message, str):
            message_bytes = message.encode("utf-8")
        else:
            message_bytes = struct.pack(f"!{len(message)}f", *message)

        # --- final packet ---
        packet = header_byte + username_bytes + message_bytes

        udp.sendto(packet, (self.server_host, self.server_udp_port))
