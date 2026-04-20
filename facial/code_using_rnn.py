import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# 1. LOAD FER2013 (48x48 → 2D SEQUENCES)
# ============================================
df = pd.read_csv("..\\fer2013\\fer2013.csv")

x, y = [], []
for row in df.itertuples():
    pixels = np.fromstring(row.pixels, dtype=np.uint8, sep=' ')
    if pixels.size == 48*48:
        x.append(pixels.reshape(48,48))  # Shape: (48 timesteps, 48 features)
        y.append(row.emotion)

x = np.array(x, dtype=np.float32) / 255.0
y = np.array(y, dtype=np.int64)

# ============================================
# 2. TRAIN/VAL/TEST SPLIT (60/20/20)
# ============================================
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# ============================================
# 3. LSTM RNN ARCHITECTURE (Stacked LSTM 128→128→256→7)
# ============================================
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(48, 48)),
    BatchNormalization(),
    Dropout(0.3),
    
    LSTM(128),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(256, activation='relu'),
    Dropout(0.4),
    
    Dense(7, activation='softmax')
])

model.summary()

# ============================================
# 4. COMPILE ADAM OPTIMIZER
# ============================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================
# 5. CALLBACKS (EARLY STOP + LEARNING RATE REDUCTION)
# ============================================
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-6)

# ============================================
# 6. TRAIN LSTM RNN (40 EPOCHS MAX)
# ============================================
history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=40,
    validation_data=(x_val, y_val),
    callbacks=[early_stop, lr_reduce]
)

# ============================================
# 7. FINAL TEST EVALUATION
# ============================================
loss, accuracy = model.evaluate(x_test, y_test)
print(f"
✅ LSTM RNN Test Accuracy: {accuracy*100:.2f}%")

# ============================================
# 8. GENERATE PREDICTIONS
# ============================================
y_pred = np.argmax(model.predict(x_test), axis=1)

# ============================================
# 9. CONFUSION MATRIX HEATMAP
# ============================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("LSTM RNN Confusion Matrix - FER2013")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ============================================
# 10. CLASSIFICATION REPORT (PER-CLASS METRICS)
# ============================================
print("
📊 LSTM RNN CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(y_test, y_pred, digits=4, zero_division=1))

# ============================================
# 11. TRAINING HISTORY PLOT (ACC/LOSS CURVES)
# ============================================
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ============================================
# 12. SAVE LSTM MODEL
# ============================================
model.save("emotion_rnn_model.keras")
print("✅ LSTM RNN model saved successfully")
