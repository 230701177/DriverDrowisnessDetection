import cv2  # type: ignore
import os
import numpy as np

data = []
labels = []

hog = cv2.HOGDescriptor()

classes = ["Drowsy", "Non Drowsy"]

# Step 1: Count total images
total_images = 0
for cls in classes:
    folder = f"processed_dataset1/{cls}"
    if os.path.exists(folder):
        total_images += len(os.listdir(folder))

processed = 0

# Step 2: Extract features
for label, cls in enumerate(classes):
    folder = f"processed_dataset1/{cls}"

    if not os.path.exists(folder):
        print(f"Warning: Folder not found -> {folder}")
        continue

    for img_name in os.listdir(folder):
        path = os.path.join(folder, img_name)

        img = cv2.imread(path, 0)

        if img is None:
            continue

        try:
            features = hog.compute(img)

            if features is None:
                continue

            data.append(features.flatten())
            labels.append(label)

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue

        # Update progress
        processed += 1
        progress = (processed / total_images) * 100
        print(f"Extracting: {progress:.2f}% ({processed}/{total_images})", end="\r")

print("\nFeature Extraction Done ✅")

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Save files
np.save("features.npy", data)
np.save("labels.npy", labels)

print("Features and Labels Saved ✅")