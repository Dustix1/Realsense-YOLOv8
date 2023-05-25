import numpy as np
import cv2
import time
from datetime import datetime
import sys
import enum
from typing import List
import argparse
import UdpComms as U
import pyrealsense2 as rs
import torch
from ultralytics import YOLO

host, port = "127.0.0.1", 8007


class PoseLandmark(enum.IntEnum):
  """The 33 pose landmarks."""
  NOSE = 0
  LEFT_EYE_INNER = 1
  LEFT_EYE = 2
  LEFT_EYE_OUTER = 3
  RIGHT_EYE_INNER = 4
  RIGHT_EYE = 5
  RIGHT_EYE_OUTER = 6
  LEFT_EAR = 7
  RIGHT_EAR = 8
  MOUTH_LEFT = 9
  MOUTH_RIGHT = 10
  LEFT_SHOULDER = 11
  RIGHT_SHOULDER = 12
  LEFT_ELBOW = 13
  RIGHT_ELBOW = 14
  LEFT_WRIST = 15
  RIGHT_WRIST = 16
  LEFT_PINKY = 17
  RIGHT_PINKY = 18
  LEFT_INDEX = 19
  RIGHT_INDEX = 20
  LEFT_THUMB = 21
  RIGHT_THUMB = 22
  LEFT_HIP = 23
  RIGHT_HIP = 24
  LEFT_KNEE = 25
  RIGHT_KNEE = 26
  LEFT_ANKLE = 27
  RIGHT_ANKLE = 28
  LEFT_HEEL = 29
  RIGHT_HEEL = 30
  LEFT_FOOT_INDEX = 31
  RIGHT_FOOT_INDEX = 32

class AngleLandmarks:
    combinations = {
        "RIGHT_KNEE" : (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_HIP),
        "LEFT_KNEE" : (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_HIP),
        "RIGHT_ELBOW" : (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_SHOULDER),
        "LEFT_ELBOW" : (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_SHOULDER),
        #"RIGHT_HIP_SHOULDER" : (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_SHOULDER),
        #"LEFT_HIP_SHOULDER" : (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_SHOULDER),
        #"RIGHT_SHOULDER" : (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_SHOULDER, PoseLandmark.LEFT_SHOULDER),
        #"LEFT_SHOULDER" : (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER)
    }

class OnePoint:
    def __init__(self, index = 0, x = 0, y = 0, z_mpipe = 0.0, visibility = 0.0, z_depth = 0, x_global = 0.0, y_global = 0.0, z_global = 0.0, x_window = 0.0, y_window = 0.0):
        self.index = index
        self.x = x
        self.y = y
        self.z_mpipe = z_mpipe
        self.visibility = visibility
        self.z_depth = z_depth
        self.x_global = x_global
        self.y_global = y_global
        self.z_global = z_global
        self.x_window = x_window
        self.y_window = y_window
    def __str__(self):
        return "{}; {}; {}; {:.5f}; {:.5f}".format(self.index, self.x_window, self.y_window, self.z_depth, self.visibility)
    def return_np_array(self, use_values : str = "zd"):
        if use_values == "zp":
            return np.array([self.x, self.y, self.z_mpipe])
        elif use_values == "zd":
            return np.array([self.x, self.y, self.z_depth])
        elif use_values == "global":
            return np.array([self.x_global, self.y_global, self.z_global])

def compute_angle(points : List[OnePoint], indexs, use_values : str = "zd"):
    vector_a = points[indexs[1]].return_np_array(use_values) - points[indexs[0]].return_np_array(use_values)
    vector_b = points[indexs[1]].return_np_array(use_values) - points[indexs[2]].return_np_array(use_values)

    vector_a = vector_a / np.sqrt(np.sum(vector_a ** 2))
    vector_b = vector_b / np.sqrt(np.sum(vector_b ** 2))

    return np.arccos(vector_a.dot(vector_b) / np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        
def online_run(stdout = False, no_window = False, offline_run = False):

    #sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #sock = U.UdpComms(udpIP="127.0.0.1", portTX=8002, portRX=8003, enableRX=True, suppressWarnings=True)
    #sock = U.UdpComms(udpIP="85.70.252.121", portTX=82, portRX=8008, enableRX=False, suppressWarnings=True)
    sock = U.UdpComms(udpIP=host, portTX=port, portRX=8008, enableRX=False, suppressWarnings=True)
    #sock.connect((host,port))

    device = torch.device('cuda:0')
    rgb_images = []
    depth_images = []
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    pipeline.start(config)

    # Load YOLOv8 models
    pose_model = YOLO('yolov8x-pose-p6.pt')  # load the pose estimation model

    # Get class labels
    pose_classes = pose_model.module.names if hasattr(pose_model, 'module') else pose_model.names

    pose_colors = np.random.uniform(0, 255, size=(len(pose_classes), 3))

    #if not stdout:
        #stdscr = curses.initscr()
        #curses.noecho()
        #stdscr.clear()
    
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
            pose_results = pose_model(color_image, device=device)
            pose_color_image = pose_results[0].plot()

            print(pose_results)
            
            if not stdout:
                #print("")
                data = ""
                if (len(point_list)==33):
                    for point in point_list:
                        data += str(point.index) + ";" + str(point.x_global) + ";" + str(point.y_global)+ ";" + str(point.z_global) + ";"
                    data+= strAngles
                    data = data[:-1]
                    file1 = open("motionCapture2.txt", 'a')
                    file1.write(data+"\n")
                    file1.close
                    sock.SendData(data)
                #stdscr.refresh()
                
                # Calculate FPS every second
                if time.time() - start_time >= 1:
                    fps = frame_count / (time.time() - start_time)
                    start_time = time.time()
                    frame_count = 0

                # Display FPS on the output image
                cv2.putText(pose_color_image, f"FPS: {fps:.2f}", (10, 680), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)         
                
                # Display the output image
                cv2.imshow('Pose estimation', pose_color_image)
                #cv2.resizeWindow('Combined Output with Depth', 1920, 1080)

                # Increment frame count
                frame_count += 1

                # Break the loop if 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        # Stop streaming
        pipeline.stop()
        
if __name__ == "__main__":
    #print('Zadejte IP adresu: ')
    #host = input()
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stdout', help="Output data to stdout", action='store_true')
    parser.add_argument('-n', '--no_window', help="Run without draw windows", action='store_true')
    parser.add_argument('-o', '--offline_run', help="Run offline from folder", action='store_true')
    parsed_arguments = vars(parser.parse_args(sys.argv[1:]))

    online_run(parsed_arguments['stdout'], parsed_arguments['no_window'], parsed_arguments['offline_run'])                  
