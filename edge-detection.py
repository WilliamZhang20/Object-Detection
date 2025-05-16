import os
import cv2
from dask import delayed, compute

INPUT_DIR = "images/"
OUTPUT_DIR = "output/"
PREFIX = "edge-filtered-"

os.makedirs(OUTPUT_DIR, exist_ok=True)

@delayed
def edge_filter(filename):
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, PREFIX+filename)

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        return f"Skipped, error in {filename}"
    
    # Grayscale for smaller scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sharpened with Gaussian Blur
    sharp = cv2.GaussianBlur(gray, (3,3), 0)

    # Sobel edge detection
    sobelx = cv2.Sobel(sharp, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
    sobely = cv2.Sobel(sharp, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges
    
    # Compute gradient magnitude
    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    
    # Convert to uint8
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    cv2.imwrite(output_path, gradient_magnitude)

    return f"Done with {filename}"

# Get all valid images
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".png"))]

# Build delayed task graph
tasks = [edge_filter(f) for f in image_files]

# Run all tasks in parallel
results = compute(*tasks)

# Print summary
print("\n".join(results))