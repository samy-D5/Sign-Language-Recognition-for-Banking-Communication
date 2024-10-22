import os
import cv2

# Directory where data will be saved
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes and dataset size
number_of_classes = 5  # Changed from 3 to 5 to include 'U' and 'V'
dataset_size = 100

# Initialize the camera
cap = cv2.VideoCapture(0)

# Check if the camera was opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Loop over each class to collect data
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Display instructions before starting data collection
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        cv2.putText(frame, 'Ready? Press "Q" to start collecting data!', (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            break

    # Data collection loop for each class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        cv2.imshow('frame', frame)

        # Save the captured frame in the class-specific directory
        file_path = os.path.join(DATA_DIR, str(j), f'{counter}.jpg')
        cv2.imwrite(file_path, frame)
        print(f'Saved {file_path}')

        counter += 1

        # Stop early if 'q' is pressed during data collection
        if cv2.waitKey(25) == ord('q'):
            print('Stopping data collection early.')
            break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
