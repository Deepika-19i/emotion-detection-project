# ==============================
# STEP 1: IMPORT LIBRARIES
# ==============================
import numpy as np
import random
import tensorflow as tf  # ✅ import tensorflow first

# set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



# ==============================
# STEP 2: LOAD DATA
# ==============================
X = np.load("X_ravdess.npy")
y = np.load("y_ravdess.npy")

print("✅ Data loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)

# ==============================
# STEP 3: LABEL ENCODING
# ==============================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

print("✅ Labels encoded")
print("Emotion classes:", encoder.classes_)

# ==============================
# STEP 4: RESHAPE FOR CNN
# (1D CNN expects 3D: samples x timesteps x channels)
# ==============================
X = X[..., np.newaxis]
print("✅ Data reshaped for CNN:", X.shape)

# ==============================
# STEP 5: TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y_cat
)

print("✅ Train-test split done")
print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# ==============================
# STEP 6: BUILD 1D CNN MODEL
# ==============================
model = Sequential([
    Conv1D(64, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)),
    Conv1D(64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(128, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==============================
# STEP 7: TRAIN MODEL
# ==============================
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau

# add callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    ModelCheckpoint("cnn1d_best.h5", save_best_only=True)
]
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=32,
    epochs=25,
    callbacks=callbacks
)

# ==============================
# STEP 8: EVALUATE MODEL
# ==============================
loss, acc = model.evaluate(X_test, y_test)
print(f"\n🎯 Test Accuracy: {acc*100:.2f}%") 

