import pandas as pd
import numpy as np
import os
from PIL import Image

df = pd.read_csv("..\\fer2013\\fer2013.csv")
required_columns = {"emotion", "pixels"}
if not required_columns.issubset(df.columns):
    raise ValueError("CSV file is missing required columns!")

print("CSV validation passed: Required columns found...........")
print("data loaded successfully")
print("total rows:", len(df))
print(df.head(5))
# print(df.tail(5))

output_dir = "..\\fer2013_images"
os.makedirs(output_dir, exist_ok=True)

emotion_names = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

invalid_emotions = df[~df["emotion"].between(0, 6)]

# validation 2
if len(invalid_emotions) > 0:
    print("Warning: Invalid emotion values found!")
else:
    print("Emotion values validation passed")

for i in range(7):
    folder_name = os.path.join(output_dir, f"{i}_{emotion_names[i]}")
    os.makedirs(folder_name, exist_ok=True)

img_size = (48, 48)

valid_images = 0
skipped_images = 0

for idx, row in df.iterrows():
    try:
        emotion = int(row["emotion"])
        pixels_str = row["pixels"]

        pixels = np.fromstring(pixels_str, dtype=np.uint8, sep=" ")

        # validation 3
        if pixels.size != img_size[0] * img_size[1]:
            skipped_images += 1
            continue

        img_array = pixels.reshape(img_size)

        emotion_folder = os.path.join(
            output_dir, f"{emotion}_{emotion_names[emotion]}"
        )
        filename = os.path.join(emotion_folder, f"img_{idx}.png")

        img = Image.fromarray(img_array, mode="L")
        img.save(filename)

        valid_images += 1

    except Exception as e:
        skipped_images += 1
        print(f"Error at row {idx}: {e}")

print("All images created in 7 emotion folders!")

print("Total rows in CSV:", len(df))
print("Images successfully created:", valid_images)
print("Images skipped:", skipped_images)

if valid_images > 0:
    print("Dataset validation SUCCESSFUL ")
else:
    print("Dataset validation FAILED ")
