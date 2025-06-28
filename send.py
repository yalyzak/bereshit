import math
import serial
import time

def lean(l1,l2):
    a = math.degrees(math.atan(l1 / l2))
    a -= 2

    # Define the map function
    def map_value(value, in_min, in_max, out_min, out_max):
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    # Example degrees for leg_r_side
    leg_r_side_degrees = 1000  # Example value for leg_r_side.degrees

    # Map pulse length
    pulse_length = map_value(leg_r_side_degrees, 500, 2500, 0, 180)
    return pulse_length,a


print(serial.Serial())
# Configure the serial portz
try:
    arduino = serial.Serial(port='COM5', baudrate=115200, timeout=1)
    time.sleep(2)  # Wait for the Arduino to initialize
    print("✅ Connected to real Arduino")
except serial.SerialException:
    class FakeArduino:
        """A fake Arduino class to simulate serial communication when the real Arduino is not connected."""

        def __init__(self, port, baudrate, timeout):
            print(f"⚠️ Fake Arduino created at {port} (Baud: {baudrate}, Timeout: {timeout})")

        def write(self, data):
            # Uncomment the line below if you want to simulate data being written.
            # print(f"📝 Fake Arduino received: {data.decode().strip()}")
            pass


        def readline(self):
            return b"Fake Arduino Response\n"

        def close(self):
            print("🔌 Fake Arduino connection closed.")
        def flush(self):
            pass

        @property
        def in_waiting(self):
            return 1  # Simulate always having one line of data to read

    arduino = FakeArduino(port='COM3', baudrate=9600, timeout=1)

def send_message(header1, header2, number):
    """Send a formatted message to the Arduino."""
    header1_str = f"{header1:02}"  # Format the first header with leading zeros
    header2_str = f"{header2:02}"  # Format the second header with leading zeros
    number_str = f"{number:04}"  # Format the number with leading zeros
    message = f"{header1_str}{header2_str}{number_str}"  # Combine headers and number
    arduino.write((message + "\n").encode())  # Send message with a newline character
    # arduino.flush()
    # time.sleep(0.05)
    receive_and_print()

    # print(f"Sent: {message}")



def receive_and_print():
    """Read and print the received data from the Arduino for 0.01 seconds."""
    start_time = time.time()  # Record the start time

    while time.time() - start_time < 0.005:  # Run for 0.01 seconds
        if arduino.in_waiting > 0:  # Check if there's data available
            message = arduino.readline().decode('utf-8').strip()  # Read and decode the message
            if len(message) >= 6:  # Ensure the message is at least long enough
                header1 = message[0:2]  # First 2 characters are the first header
                header2 = message[2:4]  # Next 2 characters are the second header
                number = message[4:]  # Remaining characters are the number

                # print(f"Received Header 1: {header1}")
                # print(f"Received Header 2: {header2}")
                # print(f"Received Number: {number}")
            else:
                # print("Received malformed message:", message)
                pass
