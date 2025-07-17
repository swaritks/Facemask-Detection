import os
from datasets import load_dataset
from PIL import Image
from io import BytesIO

dataset = load_dataset("DamarJati/Face-Mask-Detection")

os.makedirs("../data/with_mask", exist_ok=True)            # Create directory for with_mask images
os.makedirs("../data/without_mask", exist_ok=True)         # Create directory for without_mask images 

split = dataset["train"]

for i, item in enumerate(split):                        # enumarate(x) creates a list of tuples of (index, value)
    img = item["image"]                                 # This stores each image in img
    label = item["label"]
    folder = "with_mask" if label == 0 else "without_mask"      # Where to save each image
    path = f"../data/{folder}/image_{i}.jpg"                    # The path to each image
    img.save(path)                                              # Saving the image inside the folder
