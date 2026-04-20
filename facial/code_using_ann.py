import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# 1. LOAD & PREPROCESS FER2013 DATASET
# ============================================
df = pd.read_csv("..\\fer2013\\fer2013.csv")

x, y = [], []
for row in df.itertuples():
    pixels = np.fromstring(row.pixels, dtype=np.uint8, sep=' ')
    if pixels.size == 48*48:
        x.append(pixels.reshape(48,48,1))
        y.append(row.emotion)

x = np.array(x, dtype=np.float32) / 255.0  # Normalize [0,1]
x = (x - np.mean(x)) / np.std(x)           # Standardize (zero mean, unit variance)
y = np.array(y, dtype=np.int64)

# ============================================
# 2. TRAIN/VAL/TEST SPLIT (60/20/20)
# ============================================
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# ============================================
# 3. DEEP ANN ARCHITECTURE (1024→512→256→7)
# ============================================
model = Sequential([
    Flatten(input_shape=(48, 48, 1)),
    GaussianNoise(0.05),  # Regularization noise
    
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(7, activation='softmax')
])

model.summary()

# ============================================
# 4. COMPILE WITH ADAM OPTIMIZER
# ============================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================
# 5. ADVANCED CALLBACKS (EARLY STOP + LR SCHEDULING)
# ============================================
early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=6, min_lr=1e-6)

# ============================================
# 6. TRAIN ANN MODEL
# ============================================
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_val, y_val),
    callbacks=[early_stop, lr_reduce]
)

# ============================================
# 7. TEST EVALUATION
# ============================================
loss, accuracy = model.evaluate(x_test, y_test)
print(f"
✅ ANN Test Accuracy: {accuracy*100:.2f}%")

# ============================================
# 8. GENERATE PREDICTIONS
# ============================================
y_pred = np.argmax(model.predict(x_test), axis=1)

# ============================================
# 9. CONFUSION MATRIX VISUALIZATION
# ============================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("ANN Confusion Matrix - FER2013")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ============================================
# 10. DETAILED CLASSIFICATION REPORT
# ============================================
print("
📊 ANN CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(y_test, y_pred, digits=4))

# ============================================
# 11. SAVE MODEL (TensorFlow SavedModel Format)
# ============================================
model.save("emotion_ann_model_savedmodel", save_format="tf")
print("✅ Model saved in SavedModel format")
