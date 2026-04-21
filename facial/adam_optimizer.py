import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback

# 1. Load & Preprocess FER2013
df = pd.read_csv("fer2013.csv")
x, y = [], []
for row in df.itertuples():
    pixels = np.fromstring(row.pixels, dtype=np.uint8, sep=' ')
    if pixels.size == 48*48:
        x.append(pixels.reshape(48,48,1))
        y.append(row.emotion)

x, y = np.array(x, dtype=np.float32)/255.0, np.array(y)

# 2. Train/Val/Test Split (60/20/20)
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# 3. Data Augmentation
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, 
                           zoom_range=0.1, horizontal_flip=True)

# 4. CNN Model
model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(48,48,1)), BatchNormalization(), MaxPooling2D(2,2),
    Conv2D(64,(3,3),activation='relu'), BatchNormalization(), MaxPooling2D(2,2),
    Conv2D(128,(3,3),activation='relu'), BatchNormalization(), MaxPooling2D(2,2),
    Flatten(), Dense(256,activation='relu'), Dropout(0.5), Dense(7,activation='softmax')
])

model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Convergence Tracker
class ConvergenceTracker(Callback):
    def on_train_begin(self, logs=None): self.train_losses, self.train_accs = [], []
    def on_epoch_end(self, epoch, logs=None): 
        self.train_losses.append(logs['loss'])
        self.train_accs.append(logs['accuracy'])

# 6. Train
history = model.fit(datagen.flow(x_train,y_train,64), epochs=30, validation_data=(x_val,y_val),
                   callbacks=[ConvergenceTracker()], verbose=1)

# 7. Results & Save
tracker = history.history
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print(f"✅ Train: {tracker['accuracy'][-1]:.1%} | Test: {test_acc:.1%}")
print(f"Loss History: {[f'{l:.3f}' for l in tracker['loss']]}")

# Save models
model.save('cnn_emotion_model.h5')
