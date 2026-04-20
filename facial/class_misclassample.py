import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

model = tf.keras.models.load_model("emotion_cnn_augmented.keras")
print("✅ Model loaded successfully")


df = pd.read_csv("..\\fer2013\\fer2013.csv")

x, y = [], []
for row in df.itertuples():
    pixels = np.fromstring(row.pixels, dtype=np.uint8, sep=' ')
    if pixels.size == 48 * 48:
        x.append(pixels.reshape(48, 48, 1))
        y.append(row.emotion)

x = np.array(x, dtype=np.float32) / 255.0
y = np.array(y, dtype=np.int64)


_, x_temp, _, y_temp = train_test_split(
    x, y, test_size=0.4, random_state=42
)

_, x_test, _, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Found {len(x_test)} images belonging to 7 classes.")
print("✅ Test generator loaded successfully")


class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print("📌 Class Names:", class_names)

y_probs = model.predict(x_test, batch_size=64)
y_pred = np.argmax(y_probs, axis=1)
y_conf = np.max(y_probs, axis=1)

print("📋 SAMPLE PREDICTIONS:\n")

for i in range(10):
    true_cls = class_names[y_test[i]]
    pred_cls = class_names[y_pred[i]]
    conf = y_conf[i]

    status = "✅ CORRECT" if y_test[i] == y_pred[i] else "❌ WRONG"

    print(f"Sample {i}")
    print(f"   True Class      : {true_cls}")
    print(f"   Predicted Class : {pred_cls}")
    print(f"   Confidence      : {conf:.4f}")
    print(f"   Status          : {status}\n")


mis_idx = np.where(y_test != y_pred)[0]
print(f"🔴 Total misclassified samples: {len(mis_idx)}")


print("🚨 HIGH-CONFIDENCE WRONG PREDICTIONS:\n")

threshold = 0.98
high_conf_wrong = [i for i in mis_idx if y_conf[i] >= threshold]

for i in high_conf_wrong[:10]:
    print(
        f"Sample {i} | "
        f"True: {class_names[y_test[i]]} | "
        f"Predicted: {class_names[y_pred[i]]} | "
        f"Confidence: {y_conf[i]:.4f}"
    )

if len(high_conf_wrong) > 0:
    print("✅ PASS: Confident wrong predictions detected successfully.")
else:
    print("⚠️ No high-confidence wrong predictions found.")

print("And the file of misclassification is done😎")
