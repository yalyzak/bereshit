import serial
import threading
import time
from bereshit.Vector3 import Vector3
from bereshit.Quaternion import Quaternion

class readData:
    def __init__(self, port='COM4', baud_rate=115200):
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        self.running = False
        self.lock = threading.Lock()

        # Latest quaternion values
        self.X = self.Y = self.Z = 0.0
        self.accuracy = 0.0

        # Base quaternion (first stable reading)
        self.base_quat = None
        self.base_captured = False

        # Parent object (set externally by Bereshit)
        self.parent = None

    # ----------------------------------------------------------------------
    def Start(self):
        """Open serial port and start background thread"""
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            print(f"[readData] Connected to {self.port}")
        except Exception as e:
            print(f"[readData] Failed to open {self.port}: {e}")

    # ----------------------------------------------------------------------
    def _read_loop(self):
        """Continuously reads quaternion data from serial"""
        while self.running:
            try:
                line = self.ser.readline().decode(errors='ignore').strip()
                if not line:
                    continue

                parts = line.split(',')
                if len(parts) >= 3:
                    with self.lock:
                        self.X = float(parts[0])
                        self.Y = float(parts[1])
                        self.Z = float(parts[2])

                        # Capture first quaternion as base
                        if not self.base_captured:
                            self.base_quat = Vector3(self.X, self.Y, self.Z)
                            self.base_captured = True
                            print(f"[readData] Base quaternion captured: {self.base_quat}")

            except Exception as e:
                print(f"[readData] Error: {e}")
                time.sleep(0.05)

    # ----------------------------------------------------------------------
    def Update(self, dt):
        """Called every frame — updates parent rotation relative to base"""
        if not self.parent or not self.base_captured:
            return

        with self.lock:
            current = (Vector3(self.X, self.Z, self.Y).magnitude() * -9.8) + 9.8
            # Compute relative rotation from base
            # relative = current + self.base_quat
            # Apply to parent
            self.parent.position.y += ((current * dt * dt) /2) * 100
            print((current * dt * dt) /2)

    # ----------------------------------------------------------------------
    def Stop(self):
        """Stops the thread and closes the serial port"""
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        print("[readData] Stopped")
