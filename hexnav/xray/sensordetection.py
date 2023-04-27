import cv2
import numpy as np

from ..constants import *


def run_inference(image, net):
    blob = cv2.dnn.blobFromImage(image, 1 / 255,  (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)
 
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    confidences = []
    boxes = []
    rows = outputs[0].shape[1]
    image_height, image_width = image.shape[:2]
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = outputs[0][0][r]
        cx, cy, w, h = row[:4]
        confidence = row[4]
        classes_scores = row[5:]
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        class_id = np.argmax(classes_scores)
        if classes_scores[class_id] < SCORE_THRESHOLD:
            continue
        left = int((cx - w / 2) * x_factor)
        top = int((cy - h / 2) * y_factor)
        width = int(w * x_factor)
        height = int(h * y_factor)
        box = np.array([left, top, width, height])
        boxes.append(box)
        confidences.append(confidence)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    return np.array(boxes)[indices], np.array(confidences)[indices]


def detect_sensor(image, arch='yolov5s'):
    """
    Detect EMT sensor (180 type) on X-ray image obtained by frame grabber.
    """
    weights = f'../data/sensor_detection/best_{arch}.onnx'
    net = cv2.dnn.readNet(weights)
    boxes, confidences = run_inference(image, net)
    if len(boxes) == 0 or len(confidences) == 0:
        return [], []
    max_confidence = np.argmax(confidences)
    return boxes[max_confidence], confidences[max_confidence]