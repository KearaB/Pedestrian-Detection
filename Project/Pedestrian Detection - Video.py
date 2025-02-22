# ⚡ Imports
import cv2
import imutils
import time
import os

# ⚡ Input & Output Paths
input_video_path = "input/video.mp4"
output_dir = "output/"
output_video_path = os.path.join(output_dir, "video_detected.mp4")

# ⚡ Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# ⚡ Check if input video exists
if not os.path.exists(input_video_path):
    print(f"❌ Error: Input video file '{input_video_path}' not found!")
    exit()
else:
    print(f"✅ Input video found: {input_video_path}")

# ⚡ Open video file
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"❌ Error: Could not open video '{input_video_path}'. Check if the file is valid.")
    exit()
else:
    print("✅ Video file opened successfully.")

# ⚡ Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ⚡ Validate video properties
if fps == 0 or frame_width == 0 or frame_height == 0:
    print("❌ Error: Invalid video properties. Check the file format.")
    cap.release()
    exit()
else:
    print(f"✅ Video properties validated: {frame_width}x{frame_height} at {fps} FPS.")

# ⚡ Define the output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    print("❌ Error: Could not initialize video writer.")
    cap.release()
    exit()
else:
    print(f"✅ Output video writer initialized: {output_video_path}")

# ⚡ Initialize HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
print("✅ HOG pedestrian detector initialized.")

# ⚡ Start timer
start_time = time.time()
frame_count = 0
print("🎥 Processing video... (Press 'q' to exit manually)")

try:
    while cap.isOpened():
        # Stop after 12 seconds
        if time.time() - start_time > 12:
            print("⏳ Stopping: 12 seconds elapsed.")
            break

        # Read the video frame
        ret, frame = cap.read()
        if not ret:
            print("✅ End of video.")
            break

        # Skip every 2nd frame for faster processing
        if frame_count % 1 == 0:
            # Resize frame while maintaining aspect ratio
            try:
                frame = imutils.resize(frame, width=min(500, frame.shape[1]))
            except Exception as e:
                print(f"⚠️ Warning: Error resizing frame - {e}")
                continue

            # Detect pedestrians
            (regions, _) = hog.detectMultiScale(
                frame, 
                winStride=(4, 4),  # Optimized step size
                padding=(8, 8),    # Increased padding for fewer false positives
                scale=1.06         # Adjusted scale for better speed/accuracy balance
            )

            # Draw bounding boxes
            for (x, y, w, h) in regions:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green boxes for visibility

            # Write the frame to output video
            out.write(frame)

            # Display the output
            cv2.imshow("Pedestrian Detection", frame)

        frame_count += 1  # Increment frame count

        # Stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("🛑 Stopping: 'q' key pressed.")
            break

except Exception as e:
    print(f"❌ Critical Error: {e}")

finally:
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Processed video saved to: {output_video_path}")
