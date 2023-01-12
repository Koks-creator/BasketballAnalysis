from dataclasses import dataclass
from typing import Tuple, List, Union
from math import hypot
import cv2
import numpy as np
from time import time
from collections import deque

from BasketBallowe.kalman_filter import KalmanFilter


@dataclass
class DetectionData:
    x: int
    y: int
    w: int
    h: int
    class_name: str
    detections_conf: float
    color: list


@dataclass
class Detector:
    weights_file_path: str
    config_file_path: str
    classes_file_path: str
    image_width: int = 416
    image_height: int = 416
    confidence_threshold: float = 0.3
    nms_threshold: float = 0.3

    def __post_init__(self):
        self.net = cv2.dnn.readNet(self.weights_file_path, self.config_file_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        with open(self.classes_file_path) as f:
            self.classes = f.read().splitlines()

        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def detect(self, img: np.array):
        """
        :param img: input img
        :return: list of tuples containing the following data: x, y, w, h, class_name, confidence, class_color
        """
        bbox = []
        class_ids = []
        confs = []

        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1/255, (self.image_width, self.image_height), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                # print(class_id)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int((detection[0] * width) - w/2)
                    y = int((detection[1] * height) - h/2)

                    bbox.append([x, y, w, h])
                    class_ids.append(class_id)
                    confs.append(float(confidence))

        indexes = cv2.dnn.NMSBoxes(bbox, confs, self.confidence_threshold, self.nms_threshold)

        detections_list = []
        for i in indexes:
            i = i[0]

            box = bbox[i]
            x, y, w, h = box
            class_name = self.classes[class_ids[i]].capitalize()
            conf = confs[i]
            class_color = [int(c) for c in self.colors[class_ids[i]]]

            detections_list.append(DetectionData(x, y, w, h, class_name, conf, class_color))

        return detections_list

