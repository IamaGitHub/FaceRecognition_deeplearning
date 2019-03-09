import dlib
import cv2
import numpy

#load detector
dlib_detector = dlib.get_frontal_face_detector()
#use CNN to detect the face which speed is so slowly without the GPU
# dlib_detector = dlib.cnn_face_detection_model_v1('./net/mmod_human_face_detector.dat')
#define the position of face
x0 = 0
y0 = 0
x1 = 0
y1 = 0

#load camera
cap = cv2.VideoCapture(0)
flag = 0
while(True):
    ret, frame = cap.read()
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect the face
    dets = dlib_detector(grayImg, 1)
    # print(dets)
    cv2.imshow('wind', grayImg)
    for i, pos in enumerate(dets):
        x0 = pos.left()
        y0 = pos.top()
        x1 = pos.right()
        y1 = pos.bottom()
        print(x0,x1,y0,y1)
    if 0!=x0 and 0!=y0 and 0!=x1 and 0!=y1 :
        print("this is {}".format(flag)+"img")
        resImg = frame[y0:y1, x0:x1]
        cv2.imwrite("./img/king_{}".format(flag)+".jpg", resImg)
        flag = flag+1

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
