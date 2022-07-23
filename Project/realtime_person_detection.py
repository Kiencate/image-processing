import cv2
import joblib
import numpy as np
import time
from Sliding_scale import sliding_window as sd
from hog1 import hog_feature



model = joblib.load("models/my_model.dat")
size = (64, 128)
step_size = (5, 10)
zoom_pixel = 10
index = 0
def check_person(img):
    if img.shape[0] == 0 or img.shape[1] == 0:
        return 0
    img = cv2.resize(img, (80, 160))
    for (x, y, window) in sd(img, size, step_size):
        if window.shape[0] != size[1] or window.shape[1] != size[0]:
            continue
        fd = hog_feature(window, cell_size=8)
        fd = fd.reshape(1, -1)
        pred = model.predict(fd)
        sc = model.decision_function(fd)
        if pred == 1 and sc > 0.5:
            return sc[0], 1
    return sc, 0


cap = cv2.VideoCapture("test/768x576.avi")
last_gray = None
idx = -1
frame = cv2.imread("test/first_frame.jpg")
fisrt_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
while(True):
    start = time.time()
    ret, frame = cap.read()  # read frames
    idx += 1
    if not ret:
        print('Stopped reading the video ')
        break
    img = frame.copy()
    # convert color image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, fisrt_frame)
    _, diff_masked = cv2.threshold(diff, 20, 255, cv2.THRESH_OTSU)
    test = diff_masked.copy()
    cv2.imshow("threshold", diff_masked)

    diff_masked = cv2.morphologyEx(
        diff_masked, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    diff_masked = cv2.morphologyEx(
        diff_masked, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, hierarchy = cv2.findContours(
        diff_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_square = 0
    x = 0
    y = 0
    h = 0
    w = 0
    num_person = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            # print(x,y,w,h)
            if x > zoom_pixel:
                x -= zoom_pixel
            else:
                x = 0
            if y > zoom_pixel:
                y -= zoom_pixel
            else:
                y = 0
            if x+w > frame.shape[1]-zoom_pixel*2:
                w = frame.shape[1]-x
            else:
                w += zoom_pixel*2
            if y+h > frame.shape[0]-zoom_pixel*2:
                h = frame.shape[0]-y
            else:
                h += zoom_pixel*2
            crop = gray[y:y+h, x:x+w].copy()
            score, predict = check_person(crop)
            if(predict == 1):
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, "person", (x, y-10),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 0, 255))
                cv2.putText(img, "sc: %.2f" % score, (x, y+10),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 255, 255))
                num_person += 1
    cv2.putText(img, f"count: {num_person}", (
        frame.shape[1]-200, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 255, 255))
    cv2.imshow("result", img)
    cv2.imshow("mask", diff_masked)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
