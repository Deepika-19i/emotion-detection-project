import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback

# ============================================
# 1. LOAD FER2013 DATASET
# ============================================
df = pd.read_csv("..\\fer2013\\fer2013.csv")

x, y = [], []
for row in df.itertuples():
    pixels = np.fromstring(row.pixels, dtype=np.uint8, sep=' ')
    if pixels.size == 48 * 48:
        x.append(pixels.reshape(48, 48, 1))
        y.append(row.emotion)

x = np.array(x, dtype=np.float32) / 255.0
y = np.array(y, dtype=np.int64)

# ============================================
# 2. TRAIN/VAL/TEST SPLIT (60/20/20)
# ============================================
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.4, random_state=42
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42
)

# ============================================
# 3. DATA AUGMENTATION
# ============================================
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# ============================================
# 4. CNN MODEL ARCHITECTURE
# ============================================
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

# ============================================
# 5. COMPILE WITH ADAM OPTIMIZER
# ============================================
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=adam,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================
# 6. TRAINING SETUP
# ============================================
EPOCHS = 30
BATCH_SIZE = 64

# ============================================
# 7. CONVERGENCE TRACKER CALLBACK
# ============================================
class ConvergenceTracker(Callback):
    def on_train_begin(self, logs=None):
        self.train_losses = []
        self.train_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_losses.append(logs['loss'])
        self.train_accuracies.append(logs['accuracy'])

tracker = ConvergenceTracker()

# ============================================
# 8. MODEL TRAINING
# ============================================
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    callbacks=[tracker],
    verbose=1
)

# ============================================
# 9. FINAL TRAINING RESULTS
# ============================================
final_train_loss = tracker.train_losses[-1]
final_train_accuracy = tracker.train_accuracies[-1]

print("
✅ FINAL TRAINING RESULTS")
print("Final Training Loss     :", final_train_loss)
print("Final Training Accuracy :", final_train_accuracy)

# ============================================
# 10. CONVERGENCE ANALYSIS
# ============================================
threshold = 0.001
patience = 3
convergence_epoch = None

for i in range(len(tracker.train_losses) - patience):
    diffs = [
        abs(tracker.train_losses[i + j] - tracker.train_losses[i + j + 1])
        for j in range(patience)
    ]
    if all(diff < threshold for diff in diffs):
        convergence_epoch = i + 1
        break

print("
🚀 CONVERGENCE SPEED")
if convergence_epoch:
    print(f"Model converged at epoch: {convergence_epoch}")
else:
    print("Model did NOT converge within given epochs")

# ============================================
# 11. TRAINING LOSS HISTORY
# ============================================
print("
📉 TRAINING LOSS AT EACH EPOCH")
for i, loss in enumerate(tracker.train_losses, start=1):
    print(f"Epoch {i:02d} → Loss: {loss:.4f}")

# ============================================
# 12. TEST EVALUATION
# ============================================
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print("
🧪 TEST RESULTS")
print("Test Accuracy :", test_accuracy)
print("Test Loss     :", test_loss)
