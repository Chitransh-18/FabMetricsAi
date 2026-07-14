import os
import numpy as np
import cv2

def create_synthetic_wafer(filename, defect_type="none"):
    # Create dark baseline silicon layout grid (224x224)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Draw uniform background wafer circle disk space
    center = (112, 112)
    radius = 100
    cv2.circle(img, center, radius, (40, 40, 40), -1)
    cv2.circle(img, center, radius, (70, 70, 70), 1) # Wafer outline boundary
    
    # Add localized geometric anomalies based on failure mode classes
    if defect_type == "scratch":
        # Draw explicit sharp high-contrast line scratches
        cv2.line(img, (50, 60), (160, 150), (0, 255, 255), 2)
    elif defect_type == "edge-ring":
        # Draw outer perimeter cluster ring errors
        cv2.circle(img, center, 92, (0, 240, 255), 1)
    elif defect_type == "donut":
        # Draw distinct interior concentric loops
        cv2.circle(img, center, 50, (0, 255, 255), 2)
    elif defect_type == "loc":
        # Draw grouped compact particle spots clusters
        cv2.circle(img, (150, 80), 8, (0, 255, 255), -1)
        cv2.circle(img, (144, 85), 5, (0, 255, 255), -1)
    elif defect_type == "random":
        # Draw scattered point variances
        cv2.circle(img, (80, 70), 4, (0, 255, 255), -1)
        cv2.circle(img, (150, 140), 3, (0, 255, 255), -1)
        cv2.line(img, (60, 150), (90, 170), (0, 255, 255), 2)

    # Save to disk target
    cv2.imwrite(filename, img)

# Generate destination path
output_dir = "test_samples"
os.makedirs(output_dir, exist_ok=True)

# Defect distribution plan mapping out exactly 10 items
defect_plan = [
    "scratch", "edge-ring", "none", "donut", "loc", 
    "none", "random", "scratch", "edge-ring", "none"
]

print(f"Generating 10 evaluation wafer maps into folder: '{output_dir}/'...")
for i, defect in enumerate(defect_plan):
    file_path = os.path.join(output_dir, f"wafer_matrix_sample_{i+1:02d}.png")
    create_synthetic_wafer(file_path, defect_type=defect)
    print(f" ➔ Created: {file_path} [Failure Class: {defect.upper()}]")

print("\nSuccess! 10 test samples ready. Upload this entire batch folder to test the system.")