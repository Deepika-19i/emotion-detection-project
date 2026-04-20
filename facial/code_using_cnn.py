import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

x = np.array(x, dtype=np.float32)/255.0
y = np.array(y, dtype=np.int64)
"epoch"
# 2. Train / Validation / Test split
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# 3. Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# 4. CNN Model
model = Sequential([
    Conv2D(32,(3,3), activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(128,(3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])
model.summary()

# 5. Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 6. Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-6)

# 7. Train
history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    epochs=25,
                    validation_data=(x_val, y_val),
                    callbacks=[early_stop, lr_reduce])

# 8. Evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print("\nTest Accuracy:", accuracy)

# 9. Predictions
y_pred = np.argmax(model.predict(x_test), axis=1)

# 10. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 11. Precision, Recall, F1-score
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, digits=4))

# 12. Specificity per class
specificity = []
for i in range(7):
    tn = np.sum(np.delete(np.delete(cm, i, 0), i, 1))
    fp = np.sum(np.delete(cm, i, 0)[:, i])
    specificity.append(tn / (tn + fp))
print("Specificity per class:", np.round(specificity,4))

# 13. Save Model
model.save("emotion_cnn_augmented.keras")
