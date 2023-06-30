import cv2
from collections import deque
import numpy as np
import os
from utils.hubconf import custom
from utils.plots import plot_one_box
import tensorflow as tf
import time

DATASET_DIR = "littering-action-detection-model/data"
SEQUENCE_LENGTH = 20
IMAGE_SIZE = 224
path_to_model = "littering-action-detection-model/charles_model/charles_model_model_loss_0.643_acc_0.848.h5"
video_path = ""
thresh = 0.8
save = True
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

# Write Video
if save:
    out_vid = cv2.VideoWriter('output.mp4',
                              cv2.VideoWriter_fourcc(*'MP4V'),
                              fps, (original_video_width, original_video_height))

# Declare a queue to store video frames.
frames_queue = deque(maxlen=SEQUENCE_LENGTH)

# Create a blank image for the detections window
detections_image = np.zeros((original_video_height, original_video_width, 3), dtype=np.uint8)

# Set window properties
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Camera Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)

# Initialize variables for inference display
inference_frames = []
num_inference_frames = 5
prev_detection = False
detection_count = 0

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
                        cv2.rectangle(detections_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 165, 255), 2)
                        cv2.putText(detections_image, "Pulutin mo kalat mo", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

                        # Store the inference frames for display
                        inference_frames.append(frame)

                        # Display the inference frames for a limited number of times
                        if len(inference_frames) <= num_inference_frames:
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
    cv2.putText(frame, time.strftime('%Y-%m-%d %H:%M:%S'), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Camera Feed', frame)

    # Show the detections window
    cv2.imshow('Detections', detections_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_reader.release()
if save:
    out_vid.release()
cv2.destroyAllWindows()
