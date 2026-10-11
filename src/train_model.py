import cv2
import os
import numpy as np

dataset_path = "data"
model_path = "models/face_model.yml"

faces = []
labels = []
label_map = {}
current_label = 0

for file in os.listdir(dataset_path):
    if file.endswith(".jpg"):
        img_path = os.path.join(dataset_path, file)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Resize to fixed size
        gray = cv2.resize(gray, (200, 200))

        name = file.split("_")[0]
        if name not in label_map:
            label_map[name] = current_label
            current_label += 1

        faces.append(gray)
        labels.append(label_map[name])

labels = np.array(labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)
recognizer.save(model_path)

print("✅ Model trained and saved!")
print("Label map:", label_map)
