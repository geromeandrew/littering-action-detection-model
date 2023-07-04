import cv2
from collections import deque
import numpy as np
import os
from utils.hubconf import custom
from utils.plots import plot_one_box
import tensorflow as tf
import time


DATASET_DIR = "littering-action-detection-model/data"
SEQUENCE_LENGTH = 30
IMAGE_SIZE = 128
path_to_model = "littering-action-detection-model/finalv2_model/finalv2_model_model_loss_0.163_acc_0.945.h5"
video_path = "littering-action-detection-model/5.mp4"
thresh = 0.9
save = False
yolov7_model_path = "littering-action-detection-model/inference.pt"
yolov7_conf = 0.2
gpu_status = True

use_webcam = True  # Set to True if using webcam input

CLASSES_LIST = sorted(os.listdir(DATASET_DIR))

# Load LRCN_model
saved_model = tf.keras.models.load_model(path_to_model, compile=False)

# YOLOv7 Model
yolov7_model = custom(path_or_model=yolov7_model_path, gpu=gpu_status)

# Web-cam or Video
if use_webcam:
    video_reader = cv2.VideoCapture(0)  # Use default webcam (index 0)
else:
    video_reader = cv2.VideoCapture(video_path)

# Get the width and height of the video.
original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_reader.get(cv2.CAP_PROP_FPS)

# Calculate the delay in milliseconds based on the original frame rate
frame_delay_ms = int(1000 / fps)

def calculate_fps(prev_time, frame_count):
    current_time = time.time()
    time_elapsed = current_time - prev_time
    fps = frame_count / time_elapsed if time_elapsed > 0 else 0
    return current_time, fps

fps_update_interval = 5  # Update the FPS display every 5 seconds
prev_fps_update_time = time.time()
frame_count = 0


# Write Video
if save:
    out_vid = cv2.VideoWriter('output.mp4',
                              cv2.VideoWriter_fourcc(*'MP4V'),
                              fps, (original_video_width, original_video_height))

# Declare a queue to store video frames.
frames_queue = deque(maxlen=SEQUENCE_LENGTH)

# Set window properties for the camera feed
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
# cv2.setWindowProperty('Camera Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Set window properties for the inference frames
cv2.namedWindow('Inference Frames', cv2.WINDOW_NORMAL)
# cv2.setWindowProperty('Inference Frames', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Initialize variables for inference display
inference_frames = []
prev_detection = False
detection_count = 0

# Create a folder for inference images if it doesn't exist
inference_images_folder = "littering-action-detection-model/inference_images"
if not os.path.exists(inference_images_folder):
    os.makedirs(inference_images_folder)

while video_reader.isOpened():
    success, frame = video_reader.read()

    if not success:
        break

    # RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    bbox_list = []
    # Action - ROI
    results = yolov7_model(frame)
    # Bounding Box
    box = results.pandas().xyxy[0]

    for i in box.index:
        xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(
            box['ymin'][i]), int(box['xmax'][i]), int(box['ymax'][i]), box['confidence'][i]
        bbox_list.append([xmin, ymin, xmax, ymax, conf])

    if len(bbox_list) > 0:
        for bbox in bbox_list:
            if bbox[4] > yolov7_conf:
                frame_roi = frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                # Resize the Frame to fixed Dimensions.
                resized_frame = cv2.resize(frame_roi, (IMAGE_SIZE, IMAGE_SIZE))

                # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
                normalized_frame = tf.keras.utils.normalize(resized_frame, axis=0, order=2)

                # Appending the pre-processed frame into the frames list.
                frames_queue.append(normalized_frame)

                # Check if the number of frames in the queue are equal to the fixed sequence length.
                if len(frames_queue) == SEQUENCE_LENGTH:

                    # Pass the normalized frames to the model and get the predicted probabilities.
                    predicted_labels_probabilities = saved_model.predict(
                        np.expand_dims(frames_queue, axis=0))[0]

                    # Get the index of class with highest probability.
                    predicted_label = np.argmax(predicted_labels_probabilities)

                    if max(predicted_labels_probabilities) > thresh:

                        # Get the class name using the retrieved index.
                        predicted_class_name = 'littering'

                        plot_one_box(
                            bbox, frame, label=predicted_class_name,
                            color=[0, 165, 255], line_thickness=2
                        )

                        # Update the detections image with the most recent detection
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 165, 255), 2)
                        cv2.putText(frame, "Pick up your trash", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

                        # Store the inference frames for display
                        inference_frames.append(frame.copy())

                        # Display the inference frames for a limited number of times
                        cv2.imshow('Inference Frames', frame)

                    else:
                        plot_one_box(
                            bbox, frame, label='Action NOT Detected',
                            color=[128, 128, 0], line_thickness=2
                        )
            else:
                print(
                    f'[INFO] Object detection confidence: {bbox[4]} is less than given Confidence: {yolov7_conf}')

    # Write Video
    if save:
        out_vid.write(frame)

    # Show the camera feed with time and date
    cv2.putText(frame, time.strftime('%Y-%m-%d %H:%M:%S'), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Calculate FPS and update the FPS display every 'fps_update_interval' seconds
    frame_count += 1
    if time.time() - prev_fps_update_time > fps_update_interval:
        prev_fps_update_time, fps = calculate_fps(prev_fps_update_time, frame_count)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        frame_count = 0

    cv2.imshow('Camera Feed', frame)

    # Wait for a short duration to control the output video's frame rate
    cv2.waitKey(frame_delay_ms)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    if len(inference_frames) > 0:
        # Save detected frames with date and time of detection as filename
        for idx, frame in enumerate(inference_frames):
            now = time.strftime('%Y%m%d_%H%M%S')
            filename = f"{inference_images_folder}/inference_{now}_{idx}.jpg"
            cv2.imwrite(filename, frame)

        # Clear the inference frames list after saving
        inference_frames.clear()

video_reader.release()
if save:
    out_vid.release()
cv2.destroyAllWindows()
