import torch
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time

device = torch.device('cuda:0')

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
#config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
#config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)

pipeline.start(config)

# Load YOLOv8 models
seg_model = YOLO('yolov8n.pt')  # load the segmentation model
pose_model = YOLO('yolov8n-pose.pt')  # load the pose estimation model

# Get class labels
seg_classes = seg_model.module.names if hasattr(seg_model, 'module') else seg_model.names
pose_classes = pose_model.module.names if hasattr(pose_model, 'module') else pose_model.names

seg_colors = np.random.uniform(0, 255, size=(len(seg_classes), 3))
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
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Perform segmentation
        seg_results = seg_model(color_image, device=device)
        seg_color_image = seg_results[0].plot()

        # Perform pose estimation
        pose_results = pose_model(color_image, device=device)
        pose_color_image = pose_results[0].plot()

        # Create a combined output image
        combined_output = np.hstack((seg_color_image, pose_color_image))
        combined_output = cv2.resize(combined_output, (1920, 540))  # Adjust the dimensions as desired

        # Convert depth image to RGB for visualization
        depth_image_rgb = cv2.cvtColor(cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        depth_image_rgb = cv2.resize(depth_image_rgb, (combined_output.shape[1], combined_output.shape[0]))

        # Create the final output image by stacking the combined output and depth image vertically
        output_image = np.vstack((combined_output, depth_image_rgb))

        # Calculate FPS every second
        if time.time() - start_time >= 1:
            fps = frame_count / (time.time() - start_time)
            start_time = time.time()
            frame_count = 0

        # Display FPS on the output image
        cv2.putText(output_image, f"FPS: {fps:.2f}", (10, 980), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Display the output image
        cv2.imshow('Combined Output with Depth', output_image)
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
