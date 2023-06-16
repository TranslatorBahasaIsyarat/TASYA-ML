import time
import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the sign language recognition model
model = tf.keras.models.load_model('model-with-transfer-learning.h5')

mphands = mp.solutions.hands
hands = mphands.Hands(max_num_hands=1)  # Set max_num_hands to 1
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

_, frame = cap.read()
h, w, c = frame.shape

label_mapping = {
    29: 'Other',  # Default label for folders other than 'A' to 'Z', 'del', 'nothing', 'space'
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: 'del', 27: 'nothing', 28: 'space'
}

while True:
    _, frame = cap.read()

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks

    if hand_landmarks:
        handLMs = hand_landmarks[0]  # Get the first hand landmark only

        x_max = 0
        y_max = 0
        x_min = w
        y_min = h
        for lm in handLMs.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y

        y_min -= 20
        y_max += 20
        x_min -= 20
        x_max += 20

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min),
                      (x_max, y_max), (0, 255, 0), 2)

        # Extract region of interest (ROI) and preprocess it
        roi = frame[y_min:y_max, x_min:x_max]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(roi_gray, (64, 64))
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2RGB)
        roi_normalized = roi_rgb / 255.0  # Normalize the ROI

        # Expand dimensions to match the model's input shape
        roi_normalized = np.expand_dims(roi_normalized, axis=0)

        # Predict the sign language gesture
        prediction = model.predict([roi_normalized, roi_normalized])  # Pass both input tensors
        predicted_label = label_mapping[np.argmax(prediction)]

        # Display the predicted label
        cv2.putText(frame, predicted_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("Sign Language Recognition", frame)

cap.release()
cv2.destroyAllWindows()
