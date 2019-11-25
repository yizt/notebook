# -*- coding: utf-8 -*-
"""
 @File    : cv_face_recognition.py
 @Time    : 2019/11/13 下午2:59
 @Author  : yizuotian
 @Description    :
"""

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join


def train_model(face_dir):
    only_files = [f for f in listdir(face_dir) if isfile(join(face_dir, f))]
    faces, ids = [], []

    for i, files in enumerate(only_files):
        image_path = face_dir + only_files[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        faces.append(np.asarray(images, dtype=np.uint8))
        ids.append(i)

        ids = np.asarray(ids, dtype=np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()  # face is in opencv-contrib-python module
    model.train(np.asarray(faces), np.asarray(ids))
    print("Model Training Complete!!!")


def face_detector(classifier, img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return img, []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


def main():
    face_classifier = cv2.CascadeClassifier(
        '/Users/yizuotian/miniconda2/envs/pytorch/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    # model = train_model('')

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        image, face = face_detector(face_classifier, frame)
        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # result = model.predict(face)
            # if result[1] < 500:
            #     confidence = int(100 * (1 - (result[1]) / 300))
            #     display_string = str(confidence) + '% Confidence it is user'
            # cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)
            #
            # if confidence > 75:
            #     cv2.putText(image, "Face Matched", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            #     cv2.imshow('Face Cropper', image)
            cv2.imshow('Face Cropper', image)

        except:
            cv2.putText(image, "Face Not Matched", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            pass
        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
