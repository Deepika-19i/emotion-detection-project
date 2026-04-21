import cv2
import numpy as np
import tensorflow as tf

# LOAD CNN MODEL
MODEL_PATH = r"C:\Users\EXCERPT 4\Desktop\multi human emotion detection\fer_2013\emotion_cnn_augmented.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Emotion CNN model loaded")

emotion_labels = [
    'angry', 'disgust', 'fear',
    'happy', 'neutral', 'sad', 'surprise'
]


# LOAD FACE DETECTOR (OpenCV)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# WEBCAM
cap = cv2.VideoCapture(0)
print("🎥 Press Q or Ctrl+C to stop")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            face = gray_frame[y:y+h, x:x+w]

            if face.size == 0:
                continue

            face_resized = cv2.resize(face, (48, 48))
            face_normalized = face_resized / 255.0
            face_input = face_normalized.reshape(1, 48, 48, 1)

            preds = model.predict(face_input, verbose=0)
            emotion = emotion_labels[np.argmax(preds)]
            confidence = np.max(preds) * 100

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(
                frame,
                f"{emotion} ({confidence:.1f}%)",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0,255,0),
                2
            )

        cv2.imshow("Real-Time Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed safely")
