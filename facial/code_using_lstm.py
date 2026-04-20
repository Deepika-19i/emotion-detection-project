import warnings
warnings.filterwarnings("ignore")  # Suppress warnings

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load FER2013 dataset
df = pd.read_csv("..\\fer2013\\fer2013.csv")

x, y = [], []
for row in df.itertuples():
    pixels = np.fromstring(row.pixels, dtype=np.uint8, sep=' ')
    if pixels.size == 48*48:
        x.append(pixels.reshape(48,48,1))
        y.append(row.emotion)

x = np.array(x, dtype=np.float32) / 255.0
y = np.array(y, dtype=np.int64)

# 2. Train / Validation / Test split
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.4, random_state=42
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42
)
# RNN expects 3D input: (samples, timesteps, features)
x_train = x_train.reshape(-1, 48, 48)
x_val   = x_val.reshape(-1, 48, 48)
x_test  = x_test.reshape(-1, 48, 48)

# 4. LSTM Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(48, 48)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(7, activation='softmax')
])

model.summary()

# 4. Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True
)

lr_reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=4,
    min_lr=1e-6
)

# 6. Train LSTM
history = model.fit(
    x_train, y_train,
    epochs=40,
    batch_size=128,
    validation_data=(x_val, y_val),
    callbacks=[early_stop, lr_reduce]
)

# 7. Evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print("\nTest Accuracy:", accuracy)

# 8. Predictions
y_pred = np.argmax(model.predict(x_test), axis=1)

# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("LSTM Confusion Matrix")
plt.show()

# 10. Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, digits=4))

# 11. Save Model
model.save("emotion_lstm_model.h5")
print("Model saved as emotion_lstm_model.h5")
