import os
import customtkinter as ctk
import numpy as np
import pickle
import librosa
import sounddevice as sd
import cv2
import speech_recognition as sr
import threading
from collections import Counter

from tensorflow.keras.models import load_model
from PIL import Image

# BASE PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def safe_load_model(path):
    try:
        return load_model(path)
    except Exception as e:
        print(f"❌ Model load failed: {path}")
        print(e)
        return None

# LOAD MODELS 
voice_model = safe_load_model(os.path.join(BASE_DIR, "models/ravdess_cnn_model.h5"))
ann_model   = safe_load_model(os.path.join(BASE_DIR, "models/ann_emotion_model.h5"))

try:
    voice_le = pickle.load(open(os.path.join(BASE_DIR, "encoders/label_encoder.pkl"), "rb"))
    vectorizer = pickle.load(open(os.path.join(BASE_DIR, "encoders/tfidf_vectorizer.pkl"), "rb"))
    ann_labels = pickle.load(open(os.path.join(BASE_DIR, "encoders/emotion_labels_ann.pkl"), "rb"))
except Exception as e:
    print("❌ Encoder loading failed:", e)

# FACE MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

face_model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(7,activation='softmax')
])

try:
    face_model.load_weights(os.path.join(BASE_DIR,"models/cnn_emotion_model.h5"))
except:
    print("❌ Face model weights missing")

face_labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

# GLOBAL 
last_voice_emotion = None
last_text_emotion  = None
last_face_emotion  = None

running = False
cap = None

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# UI
ctk.set_appearance_mode("dark")
app = ctk.CTk()
app.geometry("1000x650")
app.title("EmotionAI")

main = ctk.CTkFrame(app)
main.pack(fill="both", expand=True)

log_box = ctk.CTkTextbox(main, height=120)
log_box.pack(fill="x", padx=10, pady=10)

def log(msg):
    log_box.insert("end", msg + "\n")
    log_box.see("end")

# VOICE
def detect_voice():
    global last_voice_emotion

    if voice_model is None:
        log("❌ Voice model missing")
        return

    def run():
        try:
            log("🎤 Recording...")
            audio = sd.rec(int(22050*3), samplerate=22050, channels=1)
            sd.wait()
            audio = audio.flatten()

            mel = librosa.power_to_db(
                librosa.feature.melspectrogram(y=audio, sr=22050)
            )

            X = np.array([mel])[..., np.newaxis]

            pred = voice_model.predict(X, verbose=0)
            emotion = voice_le.inverse_transform([np.argmax(pred)])[0]

            last_voice_emotion = emotion
            log(f"🎤 Voice: {emotion}")

        except Exception as e:
            log(f"❌ Voice error: {e}")

    threading.Thread(target=run).start()

# TEXT 
def detect_text():
    global last_text_emotion

    if ann_model is None:
        log("❌ Text model missing")
        return

    text = text_entry.get()

    if not text.strip():
        log("⚠️ Enter text")
        return

    try:
        vec = vectorizer.transform([text]).toarray()
        pred = ann_model.predict(vec)[0]

        emotion = ann_labels[np.argmax(pred)]
        last_text_emotion = emotion

        log(f"📝 Text: {emotion}")

    except Exception as e:
        log(f"❌ Text error: {e}")

# SPEECH TO TEXT 
def speech_to_text():
    def run():
        r = sr.Recognizer()
        with sr.Microphone() as src:
            log("🎙 Listening...")
            audio = r.listen(src)

        try:
            t = r.recognize_google(audio)
            text_entry.delete(0,"end")
            text_entry.insert(0,t)
            detect_text()
        except:
            log("❌ Speech failed")

    threading.Thread(target=run).start()

# FACE 
def detect_face():
    global running, cap

    if cap is None:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        log("❌ Camera error")
        return

    running = True
    update_cam()

def stop_face():
    global running, cap

    running = False

    if cap:
        cap.release()
        cap = None

    cam_label.configure(text="Camera stopped")

def update_cam():
    global last_face_emotion

    if not running:
        return

    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            roi = cv2.resize(gray[y:y+h,x:x+w],(48,48))/255.0
            pred = face_model.predict(np.reshape(roi,(1,48,48,1)), verbose=0)

            emotion = face_labels[np.argmax(pred)]
            last_face_emotion = emotion

            cv2.putText(frame,emotion,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        img = img.resize((400,300))
        ctk_img = ctk.CTkImage(light_image=img, size=(400,300))

        cam_label.configure(image=ctk_img, text="")
        cam_label.image = ctk_img

    cam_label.after(30, update_cam)

# FINAL EMOTION
def final_emotion():
    emotions = []

    if last_voice_emotion:
        emotions.append(last_voice_emotion.lower())

    if last_text_emotion:
        emotions.append(last_text_emotion.lower())

    if last_face_emotion:
        emotions.append(last_face_emotion.lower())

    if not emotions:
        log("❌ No data yet")
        return

    result = Counter(emotions).most_common(1)[0][0]
    log(f"\n🔥 FINAL EMOTION: {result.upper()}")

# UI ELEMENTS
btn_frame = ctk.CTkFrame(main)
btn_frame.pack(pady=10)

ctk.CTkButton(btn_frame,text="🎤 Voice",command=detect_voice).pack(side="left",padx=5)
ctk.CTkButton(btn_frame,text="📝 Text",command=detect_text).pack(side="left",padx=5)
ctk.CTkButton(btn_frame,text="🎙 Speak",command=speech_to_text).pack(side="left",padx=5)
ctk.CTkButton(btn_frame,text="🎥 Start Cam",command=detect_face).pack(side="left",padx=5)
ctk.CTkButton(btn_frame,text="🛑 Stop Cam",command=stop_face).pack(side="left",padx=5)
ctk.CTkButton(btn_frame,text="🔥 Final Emotion",command=final_emotion).pack(side="left",padx=5)

text_entry = ctk.CTkEntry(main, width=400)
text_entry.pack(pady=10)

cam_label = ctk.CTkLabel(main, text="Camera off")
cam_label.pack(pady=10)

# RUN 
app.mainloop() 

