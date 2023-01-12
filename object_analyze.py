from dataclasses import dataclass
from typing import Tuple
from math import hypot
import cv2
import numpy as np
from collections import deque

from BasketBallowe.kalman_filter import KalmanFilter


@dataclass
class ObjectAnalyzer:
    max_len: int = 30
    dist_threshold: int = 100

    def __post_init__(self):
        self.history = deque(maxlen=self.max_len)
        self.kf = KalmanFilter()

    def get_distance(self, p1: Tuple[int, int]) -> float:
        if self.history:
            p2 = self.history[-1]
            return hypot(p1[0] - p2[0], p1[1] - p2[1])
        else:
            return 1

    def add_point(self, cx: int, cy: int) -> None:
        dist = self.get_distance((cx, cy))

        if 0 < dist < self.dist_threshold:
            self.history.append((cx, cy))

    def draw_connections(self, img: np.array, line_color: Tuple[int, int, int] = (255, 0, 2),
                         dots_color: Tuple[int, int, int] = (0, 0, 0), line_t: int = 3, dots_r: int = 5) -> np.array:
        if len(self.history) > 1:
            for index in range(len(self.history) - 1):
                c1 = self.history[index]
                c2 = self.history[index + 1]

                cv2.line(img, c1, c2, line_color, line_t)

        for point in self.history:
            cv2.circle(img, point, dots_r, dots_color, -1)

        return img

    def draw_prediction_line(self, img: np.array, line_len: int = 5, line_color: Tuple[int, int, int] = (255, 255, 120),
                             display_info: bool = True, display_pos: Tuple[int, int] = (220, 35), font_size: int = 2,
                             font_t: int = 2) -> np.array:

        for index, ppoint in enumerate(self.history):
            first_prediction = self.kf.Estimate(ppoint[0], ppoint[1])

            if index == len(self.history) - 1:
                p_prediction = first_prediction

                # Drawing extended prediction line
                for i in range(line_len):
                    new_prediction = self.kf.Estimate(p_prediction[0], p_prediction[1])
                    if len(self.history) > line_len + 1:
                        cv2.line(img, p_prediction, new_prediction, line_color, 3)
                    else:
                        if display_info:
                            cv2.putText(img, f"Making prediction line...", display_pos, cv2.FONT_HERSHEY_PLAIN,
                                        font_size, (0, 0, 200), font_t)

                    p_prediction = new_prediction

        return img

    def clean_history(self) -> None:
        if self.history:
            self.history.clear()

    def check_if_point(self, region: np.array):
        if self.history:
            cp = self.history[-1]

            result = cv2.pointPolygonTest(region, cp, False)  # The function returns +1, -1, or 0 to indicate if a point is
                                                              # inside, outside, or on the contour
            return result