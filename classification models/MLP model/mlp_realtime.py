import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import os

# 1. Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 2. Load MLP Model and Preprocessing Tools
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(os.path.join(SCRIPT_DIR, 'mlp_model.h5'))

with open(os.path.join(SCRIPT_DIR, '../labels.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)

with open(os.path.join(SCRIPT_DIR, '../scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# 3. Initialize Webcam
cap = cv2.VideoCapture(0)

print("MLP Real-Time Demo Started. Press 'q' to quit.")

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Flip the image horizontally for a later selfie-view display
        # Convert the BGR image to RGB.
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the hand landmarks
        results = hands.process(image_rgb)

        # Draw the hand annotations on the image.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmark coordinates
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # Convert to numpy array and normalize
                landmarks = np.array(landmarks).reshape(1, -1)
                landmarks_scaled = scaler.transform(landmarks)

                # Prediction
                prediction = model(landmarks_scaled, training=False).numpy()
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)
                label = label_encoder.inverse_transform([class_id])[0]

                # Draw landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display Prediction
                text = f"{label} ({confidence*100:.1f}%)"
                cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Final Display
        cv2.imshow('MLP Gesture Recognition', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
