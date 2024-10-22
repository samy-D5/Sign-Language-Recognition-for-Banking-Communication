import serial
import time

# Initialize serial communication with Arduino
arduino_port = 'COM5'  # Change this to your Arduino's port
baud_rate = 9600  # Baud rate must match the Arduino code
ser = serial.Serial(arduino_port, baud_rate)
time.sleep(2)  # Wait for the connection to initialize

try:
    while True:
        # Replace with your logic for obtaining the prediction
        # Example: This should be your predicted letter from the hand sign recognition model
        predicted_letter = 'A'  # Replace with actual prediction logic

        # Send the letter to Arduino
        ser.write(predicted_letter.encode('utf-8'))
        time.sleep(1)  # Wait a bit before sending the next letter for stability

except KeyboardInterrupt:
    print("Program terminated by user.")
finally:
    ser.close()
    print("Serial connection closed.")
