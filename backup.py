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

# Load YOLOv5
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#model.conf = 0.5  # Set confidence threshold

# Load a model YOLOv8
model = YOLO('yolov8n-pose.pt')  # load an official model

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

        # Process the results
        pred = results.pred[0]
        class_ids = pred[:, 5].cpu().numpy().astype(int)
        confidences = pred[:, 4].cpu().numpy()
        boxes = pred[:, :4].cpu().numpy()

        # Apply non-maximum suppression to remove redundant overlapping boxes
        keep_indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        keep_indices = np.array(keep_indices).flatten()

        # Draw bounding boxes and labels on the color image
        for i in keep_indices:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]

            color = (255, 0, 0)  # Choose a color for the bounding box
            cv2.rectangle(color_image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            cv2.putText(color_image, f"{label} {confidence:.2f}", (int(x), int(y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Get keypoints from the YOLOv5 results
        keypoints = pred[:, 6:39].cpu().numpy().reshape(-1, 2)
        print(keypoints)

        # Define the bone connections
        connections = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                       (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

        # Draw bones on the color and depth images
        for connection in connections:
            start_idx, end_idx = connection
            start_keypoint = keypoints[start_idx] if start_idx < len(keypoints) else None
            end_keypoint = keypoints[end_idx] if end_idx < len(keypoints) else None

            # Check if both keypoints are detected
            if start_keypoint is not None and end_keypoint is not None:
                # Convert keypoints to pixel coordinates
                start_x, start_y = int(start_keypoint[0] * color_image.shape[1]), int(
                    start_keypoint[1] * color_image.shape[0])
                end_x, end_y = int(end_keypoint[0] * color_image.shape[1]), int(end_keypoint[1] * color_image.shape[0])

                # Draw a line between the keypoints on the color image
                color = (0, 255, 0)  # Choose a color for the bone
                cv2.line(color_image, (start_x, start_y), (end_x, end_y), color, 2)

                # Calculate the average depth between the keypoints
                depth_values = depth_image[int(start_y):int(end_y), int(start_x):int(end_x)]
                average_depth = np.mean(depth_values)

                # Draw the average depth value as text on the depth image
                text = f"Depth: {average_depth:.2f}m"
                cv2.putText(color_image, text, (start_x, start_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the output images
        cv2.imshow('Color Image', color_image)
        #cv2.imshow('Depth Image', depth_image)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
