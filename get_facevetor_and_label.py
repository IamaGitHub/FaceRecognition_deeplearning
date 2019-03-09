import cv2
import dlib
import numpy as np
import json
import os

img_path = 'D://PycharmWorkspace/pyopencvLearning/dlib_opencv_recognition_face_withCNN/img/'
data = np.zeros((1, 128))
label = []

#load face key point detector
sp = dlib.shape_predictor('./net/shape_predictor_68_face_landmarks.dat')
face_vector_extractor = dlib.face_recognition_model_v1('./net/dlib_face_recognition_resnet_model_v1.dat')

#scan the dir and load image
for file in os.listdir(img_path):
    print("filename {}".format(file))
    file_name = file
    label_name = file.split('_')[0]
    img = cv2.imread(img_path+file_name)
    h, w, c = img.shape
    rec = dlib.rectangle(0, 0, w, h)
    shape = sp(img, rec)
    face_decribe = face_vector_extractor.compute_face_descriptor(img, shape)
    face_array = np.array(face_decribe).reshape((1, 128))
    data = np.concatenate((data, face_array))
    label.append(label_name)
data = data[1:, :]
np.savetxt('./facedata.txt', data, fmt='%f')
label_file = open("./label.txt", 'w')
json.dump(label, label_file)
label_file.close()
