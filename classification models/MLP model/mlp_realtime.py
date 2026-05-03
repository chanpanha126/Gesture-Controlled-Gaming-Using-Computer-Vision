import cv2
import numpy as np
import tensorflow as tf
import pickle
import os
import time
import urllib.request
import mediapipe as mp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_TASK_PATH = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')

if not os.path.exists(MODEL_TASK_PATH):
    print("Downloading hand landmarker model (~3MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_TASK_PATH
    )
    print("Download complete.")

model = tf.keras.models.load_model(os.path.join(SCRIPT_DIR, 'mlp_model.h5'))

with open(os.path.join(SCRIPT_DIR, '../labels.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)

with open(os.path.join(SCRIPT_DIR, '../scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]

def draw_landmarks(image, landmarks, h, w):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, points[a], points[b], (0, 255, 0), 2)
    for point in points:
        cv2.circle(image, point, 4, (0, 0, 255), -1)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_TASK_PATH),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
print("MLP Real-Time Demo Started. Press 'q' to quit.")

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        h, w = image.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                draw_landmarks(image, hand_landmarks, h, w)

                landmarks = []
                for lm in hand_landmarks:
                    landmarks.extend([lm.x, lm.y, lm.z])

                landmarks_scaled = scaler.transform(np.array(landmarks).reshape(1, -1))
                prediction = model.predict(landmarks_scaled, verbose=0)
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)
                label = label_encoder.inverse_transform([class_id])[0]

                cv2.putText(image, f"{label} ({confidence*100:.1f}%)", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('MLP Gesture Recognition', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
