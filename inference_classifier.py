import cv2
import numpy as np
import pickle
import mediapipe as mp
import serial
import time
from collections import deque
import pyttsx3
import threading

# Initialize the Text-to-Speech engine
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Load the trained model
with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       min_detection_confidence=0.75,
                       min_tracking_confidence=0.75)

# Label mapping (numbers to letters)
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

# Start capturing video
cap = cv2.VideoCapture(0)

# Initialize serial communication with Arduino
arduino_port = 'COM5'  # Change this to your Arduino's port
baud_rate = 9600

# Attempt to connect to Arduino
try:
    ser = serial.Serial(arduino_port, baud_rate)
    time.sleep(2)  # Wait for the connection to initialize
    print("Arduino connected successfully.")
except serial.SerialException:
    print("Arduino not connected. Skipping serial communication.")
    ser = None  # No serial connection

# A deque to store the last 5 predictions for smoothing
predictions_queue = deque(maxlen=5)

frame_counter = 0
process_interval = 5  # Process every 5th frame
last_spoken_command = ""

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        if frame_counter % process_interval != 0:
            continue

        # Convert frame to RGB for MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = [lm.x for lm in hand_landmarks.landmark] + [lm.y for lm in hand_landmarks.landmark]

                if len(landmark_list) == 42:
                    prediction = model.predict([landmark_list])
                    predicted_letter = prediction[0]

                    predictions_queue.append(predicted_letter)
                    most_frequent_prediction = max(set(predictions_queue), key=predictions_queue.count)

                    # Map the prediction to a banking command
                    command = ''
                    if most_frequent_prediction == 'A':
                        command = 'Deposit'
                    elif most_frequent_prediction == 'B':
                        command = 'Withdraw'
                    elif most_frequent_prediction == 'L':
                        command = 'Balance Check'

                    # Send the command to Arduino
                    if ser and command:
                        ser.write((command + '\n').encode('utf-8'))  # Send with newline character
                        print(f"Sent to Arduino: {command}")

                    # Speak the command if it's different from the last spoken one
                    if command and command != last_spoken_command:
                        tts_thread = threading.Thread(target=speak_text, args=(command,))
                        tts_thread.start()
                        last_spoken_command = command

                    # Display the predicted command on the frame
                    cv2.putText(frame, f"Command: {command}",
                                (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the frame
        cv2.imshow('Banking Command Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program terminated by user.")
finally:
    cap.release()
    if ser:
        ser.close()
    cv2.destroyAllWindows()
    print("Serial connection closed.")
