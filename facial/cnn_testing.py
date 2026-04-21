import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#load trained model 
MODEL_PATH = r"C:UsersEXCERPT 4Desktopmulti human emotion detection\fer_2013emotion_cnn_augmented.keras"
model = load_model(MODEL_PATH)
print("✅ CNN model loaded successfully")

# 2. LOAD FER2013 DATASET
df = pd.read_csv("..\\fer2013\\fer2013.csv")

x, y = [], []
for row in df.itertuples():
    pixels = np.fromstring(row.pixels, dtype=np.uint8, sep=' ')
    if pixels.size == 48*48:
        x.append(pixels.reshape(48,48,1))
        y.append(row.emotion)

x = np.array(x, dtype=np.float32) / 255.0
y = np.array(y, dtype=np.int64)

# 3. EXTRACT TEST SET (20% split)
_, x_temp, _, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
_, x_test, _, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

print("Test data shape:", x_test.shape)

# 4. MODEL EVALUATION
loss, accuracy = model.evaluate(x_test, y_test)
print(f"
✅ Test Accuracy: {accuracy*100:.2f}%")

# 5. GENERATE PREDICTIONS
y_pred = np.argmax(model.predict(x_test), axis=1)

# 6. CONFUSION MATRIX VISUALIZATION
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("CNN Confusion Matrix - FER2013")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 7. DETAILED CLASSIFICATION REPORT
emotion_labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

print("📊 CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(
    y_test,
    y_pred,
    target_names=emotion_labels,
    digits=4,
    zero_division=0
))
