import cv2
import numpy as np
from time import time

from BasketBallowe.detector import Detector
from BasketBallowe.object_analyze import ObjectAnalyzer

detector = Detector(
    weights_file_path="modelBasketAndBall2/yolov3_training_final.weights",
    config_file_path="modelBasketAndBall/yolov3_testing.cfg",
    classes_file_path="modelBasketAndBall/classes.txt",
    confidence_threshold=.1,
    nms_threshold=.4
)

oa = ObjectAnalyzer(max_len=10)

BBOX_THICKNESS = 2
BBOX_FONT_SCALE = 1.5
BBOX_FONT_THICKNESS = 2
CLEAN_HISTORY_THRESH = 5
HISTORY_POINTS_COLOR = (0, 0, 200)
HISTORY_LINE_COLOR = (0, 0, 0)
HISTORY_DOTS_RADIUS = 4
POINT_THRESH = 2

cap = cv2.VideoCapture("Videos/lol2.mp4")
p_time = 0
frame_count = 0
points = 0
point_step = 0
basket_lock = False
basket_line_color = (0, 0, 200)

while cap.isOpened():
    success, frame = cap.read()

    detections = detector.detect(frame)
    detected_classes = [dt.class_name for dt in detections]

    if frame_count % CLEAN_HISTORY_THRESH == 0 and "Basket ball" not in detected_classes:
        oa.clean_history()

    for detection in detections:
        x1, y1 = detection.x, detection.y
        x2, y2 = detection.x + detection.w, detection.y + detection.h
        cx, cy = int(x1 + detection.w // 2), int(y1 + detection.h // 2)

        if detection.class_name == "Basket ball":
            oa.add_point(cx, cy)

            cv2.rectangle(frame, (x1, y1), (x2, y2), detection.color, BBOX_THICKNESS)
            frame = oa.draw_prediction_line(frame, 5)
            frame = oa.draw_connections(frame, dots_color=HISTORY_POINTS_COLOR, line_color=HISTORY_LINE_COLOR,
                                        dots_r=HISTORY_DOTS_RADIUS)

        if detection.class_name == "Basket":
            basket_region = np.array([[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]])

            cv2.polylines(frame, basket_region, True, basket_line_color, 2)
            res = oa.check_if_point(basket_region)
            if res == 1:
                point_step += 1

                if point_step == POINT_THRESH and basket_lock is False:
                    points += 1
                    point_step = 0
                    basket_lock = True
                    basket_line_color = (0, 200, 0)

            elif res == -1:
                point_step = 0
                basket_lock = False
                basket_line_color = (0, 0, 200)

        cv2.putText(frame, f"{detection.class_name} {int(round(detection.detections_conf, 2) * 100)}%",
                    (x1, y1 - 15), cv2.FONT_HERSHEY_PLAIN, BBOX_FONT_SCALE, detection.color,
                    BBOX_FONT_THICKNESS)

        frame_count += 1

    c_time = time()
    fps = int(1 / (c_time - p_time))
    p_time = c_time

    cv2.putText(frame, f"FPS: {fps}", (10, 35), cv2.FONT_HERSHEY_PLAIN, 2, (255, 200, 200), 2)
    cv2.putText(frame, f"Points: {points}", (10, 85), cv2.FONT_HERSHEY_PLAIN, 2, (200, 20, 80), 2)

    cv2.imshow("res", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
