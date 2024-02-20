from ultralytics import YOLO
import cv2
import numpy as np
import pytesseract
from dotenv import load_dotenv
from sort.sort import *
from util import write_csv
import os
import torch
from roboflow import Roboflow


# Load environment variables from .env
load_dotenv()

# Get the API key from the environment variables
roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")

# Check if the API key is available
if roboflow_api_key is None:
    raise ValueError("Roboflow API key is not set in the .env file.")

# Initialize Roboflow with the API key
rf = Roboflow(api_key=roboflow_api_key)

# Replace the project, version, and download details based on your use case
project = rf.workspace("saumil-patel-9auer").project("liquor-bottles-shape-and-label-detection")
dataset = project.version(3).download("yolov4")
model = project.version(3).model

def get_product(product_detection, track_ids):
    x1, y1, x2, y2, _, _ = product_detection
    product_center_x = (x1 + x2) / 2
    product_center_y = (y1 + y2) / 2

    for track_id, track in enumerate(track_ids):
        x_track1, y_track1, x_track2, y_track2, _ = track
        if x_track1 <= product_center_x <= x_track2 and y_track1 <= product_center_y <= y_track2:
            return x_track1, y_track1, x_track2, y_track2, track_id

    return -1

results = {}

mot_tracker = Sort()

# Load YOLO models with local GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
product_detector = YOLO('yolov4-products.pt', device=device)
label_detector = YOLO('yolov4-labels.pt', device=device)

# Load video
cap = cv2.VideoCapture('sample.mp4')

# Define classes for products (modify as needed)
product_classes = ['bottle', 'can', 'grocery']

# Read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}

        # Detect products
        product_detections = product_detector(frame)[0]
        product_detections_ = []

        for detection in product_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            class_name = product_classes[int(class_id)]
            product_detections_.append([x1, y1, x2, y2, score, class_name])

        # Track products
        track_ids = mot_tracker.update(np.asarray(product_detections_))

        # Detect and read labels using OCR
        for product_detection in product_detections_.copy():
            x1, y1, x2, y2, score, class_name = product_detection
            x_product1, y_product1, x_product2, y_product2, product_id = get_product(product_detection, track_ids)

            if product_id != -1:
                # Crop product label region
                label_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                # Perform BERT-based text recognition on the label crop
                label_text, confidence_score = read_product_label(label_crop)

                if product_label_complies_format(label_text):
                    results[frame_nmr][product_id] = {'product': {'bbox': [x_product1, y_product1, x_product2, y_product2],
                                                                   'class': class_name,
                                                                   'bbox_score': score},
                                                       'label': {'text': label_text, 'confidence_score': confidence_score}}

# Write results
write_csv(results, 'product_recognition_results.csv')
