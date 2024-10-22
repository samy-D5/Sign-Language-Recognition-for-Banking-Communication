import os
import pickle
import cv2
import numpy as np
import mediapipe as mp
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the directory for storing data
DATA_DIR = './data'

# Initialize lists to store data and labels
data = []
labels = []

# Map directory names to letters (updated to include 'U' and 'V')
label_map = {'0': 'A', '1': 'B', '2': 'L', '3': 'U', '4': 'V'}

for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_dir):
        continue


    # Check if it's a directory
    if not os.path.isdir(class_dir):
        continue

    for img_path in os.listdir(class_dir):
        img_path_full = os.path.join(class_dir, img_path)

        # Read the image
        img = cv2.imread(img_path_full)

        # Check if the image is loaded correctly
        if img is None:
            print(f"Warning: Could not read image {img_path_full}. Skipping...")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image using MediaPipe
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.append(lm.x)
                    landmark_list.append(lm.y)

                # Check if there are landmarks detected
                if len(landmark_list) == 42:  # Ensure landmarks count is valid
                    data.append(landmark_list)
                    labels.append(label_map[dir_])
                    print(f"Processed image {img_path_full}: {label_map[dir_]}")

try:
    img = cv2.imread(img_path_full)
except Exception as e:
    print(f"Error loading image {img_path_full}: {e}")

# Save data and labels
with open('dataset.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset created successfully!")
