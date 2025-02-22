# ⚡ Imports
import cv2
import imutils
import os

# ⚡ Define input and output paths
input_path = "input/image.png" # Path to input image
output_dir = "output/"          # Directory to save processed image
output_path = os.path.join(output_dir, "image_detected.png")  # Output image path

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

#⚡Check if input image exists
if not os.path.exists(input_path):
    print(f"❌ Error: Input image '{input_path}' not found!")
    exit()
else:
    print(f"✅ Input image found: {input_path}")

#⚡Load image with error handling
image = cv2.imread(input_path)

if image is None:
    print(f"❌ Error: Unable to load image at '{input_path}'. Check the file format.")
    exit()
else:
    print("✅ Image loaded successfully.")

#⚡Resize the image while maintaining aspect ratio
try:
    image = imutils.resize(image, width=min(400, image.shape[1]))
    print("✅ Image resized successfully.")
except Exception as e:
    print(f"⚠️ Warning: Error resizing image - {e}")
    exit()

#⚡Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
print("✅ HOG pedestrian detector initialized.")

#⚡Detect pedestrians with original parameters
try:
    (regions, _) = hog.detectMultiScale(
        image,
        winStride=(4, 4),  # Step size
        padding=(4, 4),    # Padding
        scale=1.05         # Scale factor
    )
    print(f"✅ Detection completed: {len(regions)} pedestrian(s) detected.")
except Exception as e:
    print(f"❌ Error: Detection failed - {e}")
    exit()

#⚡Draw bounding boxes around detected pedestrians
for (x, y, w, h) in regions:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding boxes
print("✅ Bounding boxes drawn.")

#⚡Save the output image
try:
    cv2.imwrite(output_path, image)
    print(f"✅ Processed image saved to: {output_path}")
except Exception as e:
    print(f"❌ Error: Failed to save image - {e}")
    exit()

#⚡Display the output image
try:
    cv2.imshow("Pedestrian Detection", image)
    print("🖼️  Displaying output image... (Press any key to close)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception as e:
    print(f"⚠️ Warning: Unable to display image - {e}")
