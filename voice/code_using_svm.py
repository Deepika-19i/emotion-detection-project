# ==============================
# STEP 1: IMPORT LIBRARIES
# ==============================
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ==============================
# STEP 2: LOAD PREPROCESSED DATA
# ==============================
X = np.load("X_ravdess.npy")
y = np.load("y_ravdess.npy")

print("✅ Data loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)

# ==============================
# STEP 3: LABEL ENCODING
# (emotion names → numbers)
# ==============================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("✅ Labels encoded")
print("Emotion classes:", encoder.classes_)

# ==============================
# STEP 4: FEATURE SCALING
# (VERY IMPORTANT FOR SVM)
# ==============================
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("✅ Features scaled")

# ==============================
# STEP 5: TRAIN–TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("✅ Data split completed")
print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# ==============================
# STEP 6: BUILD MODEL (SVM)
# ==============================
model = SVC(
    kernel="rbf",
    C=20,
    gamma="scale",
    class_weight="balanced"
)

print("✅ Model created")

# ==============================
# STEP 7: TRAIN MODEL
# ==============================
print("🔄 Training model...")
model.fit(X_train, y_train)
print("✅ Model training completed")

# ==============================
# STEP 8: TEST MODEL
# ==============================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Accuracy:", accuracy * 100, "%")

print("\n📊 Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# STEP 9: SAVE MODEL & TOOLS
# ==============================
joblib.dump(model, "voice_emotion_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n💾 Model saved:")
print("voice_emotion_model.pkl")
print("label_encoder.pkl")
print("scaler.pkl")
