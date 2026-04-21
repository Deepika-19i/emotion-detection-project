import os
import numpy as np
import sounddevice as sd
import librosa
from tensorflow.keras.models import load_model
import soundfile as sf

# PARAMETERS
EMOTIONS = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]
RECORD_SECONDS = 3
SAMPLE_RATE = 22050
MFCC_TIMESTEPS = 40
PRETRAINED_MODEL = "cnn1d_best.h5"

# FUNCTION: RECORD AUDIO
def record_audio(duration=RECORD_SECONDS, sr=SAMPLE_RATE):
    print(f"🎤 Recording for {duration}s...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    audio = np.squeeze(audio)
    print("✅ Recording done")
    return audio

# FUNCTION: EXTRACT MFCC FEATURES
def extract_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=40):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T
    mfccs = np.mean(mfccs, axis=1, keepdims=True)
    if mfccs.shape[0] < MFCC_TIMESTEPS:
        pad_width = MFCC_TIMESTEPS - mfccs.shape[0]
        mfccs = np.pad(mfccs, ((0,pad_width),(0,0)), mode='constant')
    else:
        mfccs = mfccs[:MFCC_TIMESTEPS,:]
    return mfccs 
    
# LOAD PRETRAINED MODEL
model = load_model(PRETRAINED_MODEL)
print("✅ Loaded pretrained CNN model:", PRETRAINED_MODEL)

# LIVE EMOTION PREDICTION
while True:
    input("\nPress Enter and speak your emotion for 3 seconds...")
    audio = record_audio()
    
    # Extract features
    mfcc = extract_mfcc(audio)
    mfcc = np.expand_dims(mfcc, axis=0)  # add batch dimension: (1,40,1)
    
    # Predict
    pred = model.predict(mfcc)
    emotion_index = np.argmax(pred)
    emotion_label = EMOTIONS[emotion_index]
    
    print(f"😃 Detected Emotion: {emotion_label.upper()}")

    # Optional: stop after one prediction
    stop = input("Do you want to test another emotion? (y/n): ").strip().lower()
    if stop != "y":
        break
