import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD & PREPROCESS FER2013 DATASET
df = pd.read_csv("..\\fer2013\\fer2013.csv")

x, y = [], []
for row in df.itertuples():
    pixels = np.fromstring(row.pixels, dtype=np.uint8, sep=' ')
    if pixels.size == 48 * 48:
        x.append(pixels.reshape(48, 48, 1))
        y.append(row.emotion)

x = np.array(x, dtype=np.float32) / 255.0
y = np.array(y, dtype=np.int64)

# 2. TRAIN/VAL/TEST SPLIT (60/20/20)
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# 3. DATA AUGMENTATION (Rotation/Shift/Zoom/Flip)
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# 4. CNN ARCHITECTURE (32→64→128 → 256→7)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.summary()

# 5. PARAMETER ANALYSIS (TRAINABLE COUNT)
print("🔍 LAYER-WISE PARAMETER BREAKDOWN")
print("-" * 55)

total_trainable = 0
for layer in model.layers:
    trainable = int(np.sum([tf.keras.backend.count_params(w) for w in layer.trainable_weights]))
    total_trainable += trainable
    print(f"{layer.name:20s} | Trainable: {trainable:>8d}")

print(f"✅ TOTAL TRAINABLE PARAMETERS: {total_trainable:,}")

# 6. COMPILE ADAM OPTIMIZER
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 7. CALLBACKS (EARLY STOP + LR SCHEDULING)
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-6)

# 8. TRAIN CNN (30 EPOCHS MAX, BATCH=64)
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=30,
    validation_data=(x_val, y_val),
    callbacks=[early_stop, lr_reduce]
)

# 9. FINAL TEST EVALUATION
loss, accuracy = model.evaluate(x_test, y_test)
print(f"
✅ FINAL TEST ACCURACY: {accuracy*100:.2f}%")

# 10. PREDICTIONS & CONFUSION MATRIX
y_pred = np.argmax(model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
plt.title("CNN Confusion Matrix - FER2013", fontsize=14, pad=20)
plt.xlabel("Predicted Emotion", fontsize=12)
plt.ylabel("Actual Emotion", fontsize=12)
plt.show()

# 11. DETAILED CLASSIFICATION REPORT
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
print("
📊 DETAILED CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_test, y_pred, target_names=emotion_labels, digits=4))

# 12. SAVE FINAL MODEL
model.save("emotion_cnn_augmented.keras")
print("✅ PRODUCTION MODEL SAVED: emotion_cnn_augmented.keras")
