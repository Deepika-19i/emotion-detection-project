import pandas as pd 
import numpy as np
df = pd.read_csv("..\\fer2013\\fer2013.csv")

print("data loaded success")
print("total rows:",len(df))
print(df.head(5))


import os
from PIL import Image

df = pd.read_csv("..\\fer2013\\fer2013.csv")
print("data loaded success")
print("total rows:", len(df))


output_folder = "fersome_images"
os.makedirs(output_folder, exist_ok=True)

img_size = (48, 48)

for i, row in df.iterrows():
    pixels_str = row["pixels"]

    
    pixels = np.fromstring(pixels_str, dtype=np.uint8, sep=' ')

    
    if pixels.size != img_size[0] * img_size[1]:
        print(f"Skipping row {i}: wrong size {pixels.size}")
        continue

    
    image_array = pixels.reshape(img_size)

    img = Image.fromarray(image_array, mode="L")

    img.save(os.path.join(output_folder, f"img_{i}.png"))

print("Image creation done!")
