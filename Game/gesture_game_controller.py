import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import pyautogui
import time
import json
import os

from collections import deque, Counter

# Disabling PyAutoGUI's built-in delay — timing is managed via COOLDOWN_TIME instead
pyautogui.PAUSE = 0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, '../classification models/CNN model/cnn_model.h5')
LABELS_PATH = os.path.join(BASE_DIR, '../classification models/labels.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '../classification models/scaler.pkl')
MAPPING_FILE = os.path.join(BASE_DIR, 'gesture_mapping.json')
TASK_PATH = os.path.join(BASE_DIR, '../classification models/MLP model/hand_landmarker.task')

BUFFER_SIZE = 2

SPECIAL_KEYS = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
    32: "space",
    13: "enter"
}

DEFAULT_MAPPINGS = {
    "Fist": "space",
    "Thumb Up": "up",
    "Peace": "left"
}


def save_mappings():
    with open(MAPPING_FILE, 'w') as f:
        json.dump(gesture_to_key, f)


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


def main():
    global gesture_to_key

    print("Loading model and metadata...")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, 'rb') as f:
        label_encoder = pickle.load(f)

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=TASK_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    gesture_buffer = deque(maxlen=BUFFER_SIZE)
    held_key = None
    mapping_mode = False
    current_gesture = "None"

    gesture_to_key = {}
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, 'r') as f:
            gesture_to_key = json.load(f)
    else:
        gesture_to_key = dict(DEFAULT_MAPPINGS)

    cap = cv2.VideoCapture(0)
    print("\n--- GESTURE GAME CONTROLLER ---")
    print("Press 'm' to Enter/Exit Mapping Mode")
    print("Press 'q' to Quit")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

                current_gesture = "None"
                confidence = 0

                if results.hand_landmarks:
                    for hand_landmarks in results.hand_landmarks:
                        draw_landmarks(frame, hand_landmarks, h, w)

                        landmarks = []
                        for lm in hand_landmarks:
                            landmarks.extend([lm.x, lm.y, lm.z])

                        input_data = np.array(landmarks).reshape(1, -1)
                        input_scaled = scaler.transform(input_data)
                        cnn_input = input_scaled.reshape(1, 63, 1).astype('float32')

                        # Using __call__ (model()) is faster than predict() for single samples
                        pred_tensor = model(cnn_input, training=False)
                        pred = pred_tensor.numpy()

                        idx = np.argmax(pred)
                        confidence = np.max(pred)

                        if confidence > 0.8:
                            raw_label = label_encoder.inverse_transform([idx])[0]
                            gesture_buffer.append(raw_label)
                        else:
                            gesture_buffer.append("Unrecognized")

                        current_gesture = Counter(gesture_buffer).most_common(1)[0][0]

                # UI
                mode_text = "MAPPING MODE (Press 'm' to save)" if mapping_mode else "GAMING MODE"
                cv2.putText(frame, mode_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if mapping_mode else (0, 255, 0), 2)
                cv2.putText(frame, f"Gesture: {current_gesture}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                mapped_key = gesture_to_key.get(current_gesture, "None")
                cv2.putText(frame, f"Mapped Key: {mapped_key}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # ACTIONS
                if not mapping_mode:
                    desired_key = gesture_to_key.get(current_gesture)
                    if desired_key == "None":
                        desired_key = None

                    if desired_key != held_key:
                        if held_key:
                            pyautogui.keyUp(held_key)
                        if desired_key:
                            pyautogui.keyDown(desired_key)
                        held_key = desired_key

                cv2.imshow('Gesture Game Controller', frame)
                key_hit = cv2.waitKey(1) & 0xFF

                if key_hit == ord('q'):
                    break
                elif key_hit == ord('m'):
                    mapping_mode = not mapping_mode
                    if not mapping_mode:
                        save_mappings()
                        print("Mappings Saved!")
                    else:
                        print("Mapping Mode Active. Perform a gesture and press a key to map it.")

                elif mapping_mode and current_gesture != "None" and key_hit != 255:
                    if key_hit not in [ord('m'), ord('q')]:
                        if key_hit in SPECIAL_KEYS:
                            gesture_to_key[current_gesture] = SPECIAL_KEYS[key_hit]
                            print(f"Mapped {current_gesture} to {SPECIAL_KEYS[key_hit]}")
                        else:
                            char = chr(key_hit)
                            gesture_to_key[current_gesture] = char
                            print(f"Mapped {current_gesture} to {char}")
    finally:
        if held_key:
            pyautogui.keyUp(held_key)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
