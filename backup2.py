import torch
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)

pipeline.start(config)

# Load YOLOv8
model = YOLO('yolov8x-seg.pt')  # load an official model

# Get class labels
classes = model.module.names if hasattr(model, 'module') else model.names

colors = np.random.uniform(0, 255, size=(len(classes), 3))

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Detect objects in the color image
        results = model(color_image)

        # Visualize the results on the frame
        color_image = results[0].plot()

        # Display the output images
        cv2.imshow('Color Image', color_image)
        # cv2.imshow('Depth Image', depth_image)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
