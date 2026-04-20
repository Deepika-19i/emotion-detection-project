import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ================= PATH =================
base_path = r"C:\Users\EXCERPT 4\.cache\kagglehub\datasets\debarshichanda\goemotions\versions\6\data\full_dataset"

# ================= LOAD ALL CSV FILES =================
csv_files = [
    os.path.join(base_path, f)
    for f in os.listdir(base_path)
    if f.endswith(".csv")
]

print("CSV files found:", csv_files)

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

print("Dataset shape:", df.shape)
print(df.head())

# ================= EMOTION MAPPING (7 CLASSES) =================

emotion_map = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "excitement", "gratitude", "love", "optimism", "relief", "admiration", "approval"],
    "sadness": ["sadness", "disappointment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion"],
    "neutral": ["neutral"]
}

selected_emotions = list(emotion_map.keys())

emotion_columns = df.columns.drop("text")

# ================= CREATE LABELS =================

def map_emotions(row):
    label = np.zeros(len(selected_emotions))

    for i, emotion in enumerate(selected_emotions):
        for sub in emotion_map[emotion]:
            if sub in emotion_columns and row[sub] == 1:
                label[i] = 1
    return label

labels = np.array(df.apply(map_emotions, axis=1).tolist())

texts = df["text"].values

# remove empty rows
mask = labels.sum(axis=1) > 0
texts = texts[mask]
labels = labels[mask]

print("Final samples:", len(texts))

# ================= TF-IDF =================

vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(texts)

pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

# ================= TRAIN TEST SPLIT =================

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# ================= MODEL =================

model = Sequential([
    Dense(512, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.5),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(128, activation='relu'),

    Dense(len(selected_emotions), activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# ================= TRAIN =================

model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# ================= EVALUATE =================

loss, acc = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", acc * 100)

# ================= SAVE =================

model.save("ann_emotion_model.h5")
pickle.dump(selected_emotions, open("emotion_labels.pkl", "wb"))

print("Model saved successfully!") 

