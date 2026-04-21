import numpy as np
import pickle
from tensorflow.keras.models import load_model

# LOAD MODEL 
model = load_model("ann_emotion_model.h5")

# LOAD VECTORIZER
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# LOAD LABELS 
labels = pickle.load(open("emotion_labels.pkl", "rb"))

# PREDICT FUNCTION
def predict_emotion(text):
    # transform text
    X = vectorizer.transform([text]).toarray()

    # prediction
    pred = model.predict(X)[0]

    # get best emotion
    top_index = np.argmax(pred)
    emotion = labels[top_index]

    print("\n📝 Text:", text)
    print("\n🎯 Predicted Emotion:", emotion)

    print("\n📊 Probabilities:")
    for i, label in enumerate(labels):
        print(f"{label}: {pred[i]:.4f}")

# REAL-TIME LOOP 
while True:
    text = input("\nEnter text (or type 'exit'): ")

    if text.lower() == "exit":
        break

    predict_emotion(text) 

