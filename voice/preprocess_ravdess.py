import os
import librosa
import numpy as np
from tqdm import tqdm

# ==============================
# DATASET PATH (VERY IMPORTANT)
# ==============================
DATASET_PATH = "audio_speech_actors_01-24"

# ==============================
# EMOTION MAPPING (RAVDESS)
# ==============================
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

X = []
y = []

print("🔄 Processing audio files...")

# Loop through Actor folders
for actor in os.listdir(DATASET_PATH):
    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    print(f"📂 Processing {actor}")

    # Loop through audio files
    for file in tqdm(os.listdir(actor_path)):
        if file.endswith(".wav"):
            file_path = os.path.join(actor_path, file)

            # Filename format:
            # 03-01-05-01-02-02-12.wav
            # emotion code is 3rd value
            emotion_code = file.split("-")[2]
            emotion = emotion_map.get(emotion_code)

            try:
                # Load audio
                y_audio, sr = librosa.load(file_path, duration=3, offset=0.5)

                # Normalize audio
                y_audio = librosa.util.normalize(y_audio)

                # Extract MFCCs (40 features)
                mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40)
                mfcc_mean = np.mean(mfcc.T, axis=0)

                X.append(mfcc_mean)
                y.append(emotion)

            except Exception as e:
                print("❌ Error:", file_path, e)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Save processed data
np.save("X_ravdess.npy", X)
np.save("y_ravdess.npy", y)

print("✅ Preprocessing completed")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("💾 Files saved: X_ravdess.npy, y_ravdess.npy")
