import cv2
import os

TARGET_DIR = "dataset"
dir_len = len(os.listdir(TARGET_DIR))
image_index = dir_len

# RzucanieDoKosza.mp4
# RzucanieDoKosza2.mp4
# MeczykKosza.mp4
# ChlopCoWsad.mp4
# wsadzikkoleny.mp4
# meczyk.mp4
# production ID_5192070.mp4"
cap = cv2.VideoCapture(r"Videos/wsadzisko.mp4")

while cap.isOpened():
    _, frame = cap.read()
    # frame = cv2.resize(frame, (600, 800))

    cv2.imshow("Res", frame)
    key = cv2.waitKey(100)

    if key == 27:
        break
    if key == ord("s"):
        print("d")
        cv2.imwrite(fr"{TARGET_DIR}/xkoks{image_index}.jpg", frame)
        image_index += 1

cap.release()
cv2.destroyAllWindows()
#C:\Users\table\PycharmProjects\pajtong\BasketBallowe\dataset\classes.txt
#C:\Users\table\PycharmProjects\pajtong\BasketBallowe\dataset