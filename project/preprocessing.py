import cv2
import os

input_folder = "Data"
output_folder = "processed_dataset1"

classes = ["Drowsy", "Non Drowsy"]

# Step 1: Count total images
total_images = 0
for cls in classes:
    class_path = os.path.join(input_folder, cls)
    if os.path.exists(class_path):
        total_images += len(os.listdir(class_path))

processed = 0

# Step 2: Process images
for cls in classes:
    input_path = os.path.join(input_folder, cls)
    output_path = os.path.join(output_folder, cls)

    os.makedirs(output_path, exist_ok=True)

    for img_name in os.listdir(input_path):
        path = os.path.join(input_path, img_name)

        img = cv2.imread(path)

        if img is None:
            continue

        # 1. Resize
        img = cv2.resize(img, (224, 224))

        # 2. Denoise
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # 3. Enhance (convert to grayscale + histogram equalization)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(gray)

        # Save processed image
        save_path = os.path.join(output_path, img_name)
        cv2.imwrite(save_path, img)

        # Update progress
        processed += 1
        progress = (processed / total_images) * 100
        print(f"Processing: {progress:.2f}% ({processed}/{total_images})", end="\r")

print("\nPreprocessing Done ✅")