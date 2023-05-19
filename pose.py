import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)

pipeline.start(config)

# Load YOLOv8 models
pose_model = YOLO('yolov8n-pose.pt')  # load the pose estimation model

# Get class labels
pose_classes = pose_model.module.names if hasattr(pose_model, 'module') else pose_model.names

pose_colors = np.random.uniform(0, 255, size=(len(pose_classes), 3))

# Initialize variables for FPS calculation
start_time = time.time()
frame_count = 0
fps = 0

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Perform pose estimation
        pose_results = pose_model(color_image)
        pose_color_image = pose_results[0].plot()

        # Calculate FPS every second
        if time.time() - start_time >= 1:
            fps = frame_count / (time.time() - start_time)
            start_time = time.time()
            frame_count = 0

        # Display FPS on the output image
        cv2.putText(pose_color_image, f"FPS: {fps:.2f}", (10, 980), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Display the output image
        cv2.imshow('Combined Output with Depth', pose_color_image)
        cv2.resizeWindow('Combined Output with Depth', 1920, 1080)

        # Increment frame count
        frame_count += 1

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
