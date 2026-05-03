import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

mp_draw = mp.solutions.drawing_utils

# Create dataset folder
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Ask user for label (exercise name)
label = input("Enter exercise name (e.g., open_hand, fist): ")

# CSV file path
file_path = f"dataset/{label}.csv"

# Open webcam
cap = cv2.VideoCapture(0)

data = []

print("\nPress 's' to start recording")
print("Press 'q' to quit\n")

recording = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

            if recording:
                landmarks.append(label)  # add label
                data.append(landmarks)

    cv2.putText(frame, f"Recording: {recording}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        recording = True
        print("Recording started...")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save data
if data:
    df = pd.DataFrame(data)
    df.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))
    print(f"Data saved to {file_path}")
else:
    print("No data recorded.")