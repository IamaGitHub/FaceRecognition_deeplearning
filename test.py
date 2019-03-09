import cv2
import dlib
import json
import os
import numpy as np

#load network model
# face_detector = dlib.get_frontal_face_detector()
face_detector = dlib.cnn_face_detection_model_v1('./net/mmod_human_face_detector.dat')
face_feature_point = dlib.shape_predictor('./net/shape_predictor_68_face_landmarks.dat')
face_feature_extract = dlib.face_recognition_model_v1('./net/dlib_face_recognition_resnet_model_v1.dat')
threshold = 0.53

#define the position of face of the frame
x0 = 0
y0 = 0
x1 = 0
y1 = 0

#read label file and facedata file
flabel = open('./label.txt')
with open('./label.txt', 'r') as file:
    flabel_data = json.load(file)

face_data_res = np.loadtxt('./facedata.txt')


#define the distance of real_face and corresponding vector which computation with NeareastNiberhod
def compute_distance(img_feature, flabel_datas):
    print(type(img_feature))
    temp = img_feature - face_data_res
    e = np.linalg.norm(temp, axis=1, keepdims=True)
    min_distance = e.min()
    if min_distance > threshold:
        return 'other_face'
    index = np.argmin(e)
    print("index:", index)
    return flabel_datas[index]

def recognition_process():
    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()
        # grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #find the face of frame
        find_face_ret = face_detector(frame, 1)
        print(find_face_ret)
        if len(find_face_ret):
            #get the position of face
            for i, pos in enumerate(find_face_ret):
                pos = pos.rect
                x0 = pos.left()
                y0 = pos.top()
                x1 = pos.right()
                y1 = pos.bottom()
                #convert the position to the dlib.reectangle
                res = dlib.rectangle(x0, y0, x1, y1)
                #find the key point of grayImg
                face_key_point = face_feature_point(frame, res)
                #find the face features
                face_features = face_feature_extract.compute_face_descriptor(frame, face_key_point)
                #datatype convert
                # face_features_arry = np.array(face_features).reshape((1, 128))
                #use the nearestNeiborhod compute the distance
                face_label = compute_distance(face_features, flabel_data)
                # print("face_label:", face_label)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
                cv2.putText(frame, face_label, (x0, y0-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                cv2.imshow('wind', frame)
        else:
            cv2.imshow('wind', frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognition_process()















































