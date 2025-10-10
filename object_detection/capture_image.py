import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Create output directory if it doesn't exist
output_dir = "dataset/images"
os.makedirs(output_dir, exist_ok=True)

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure the RealSense stream
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start the RealSense pipeline
pipeline.start(config)

print("📸 Press SPACE to capture an image, 'q' to quit.")

image_count = len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])  # Count existing images

try:
    while True:
        # Get frames from RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue  # Skip frame if no valid color image

        # Convert RealSense frame to OpenCV image
        color_image = np.asanyarray(color_frame.get_data())

        # Display the image
        cv2.imshow("RealSense Camera - Press SPACE to Capture", color_image)

        # Capture image on SPACE key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # SPACE key
            image_count += 1
            filename = os.path.join(output_dir, f"image_{image_count:03d}.jpg")
            cv2.imwrite(filename, color_image)
            print(f"✅ Image saved: {filename}")

        elif key == ord('q'):  # Quit on 'q' key
            break

finally:
    # Stop the RealSense pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
