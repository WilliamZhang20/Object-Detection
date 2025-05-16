import os
import cv2
from dask import delayed, compute

INPUT_DIR = "images/"
OUTPUT_DIR = "output/"
PREFIX = "face-detected-"

os.makedirs(OUTPUT_DIR, exist_ok=True)

@delayed
def face_detector(filename):
    # Keep local face_cascade variable for THREAD SAFETY

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, PREFIX+filename)

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        return f"Skipped, error in {filename}"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite(output_path, img)

    return f"Processed: {filename} (faces: {len(faces)})"

# Get all valid images
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".png"))]

# Build delayed task graph
tasks = [face_detector(f) for f in image_files]

# Run all tasks in parallel
results = compute(*tasks)

# Print summary
print("\n".join(results))