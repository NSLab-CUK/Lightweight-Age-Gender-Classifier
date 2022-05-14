'''
python = 3.7
tensorflow = 1.14.0
tensorflow-gpu = 1.14.0
cudnn = 7.6.5
cudatoolkit = 10.0.130
h5py = 2.10.0
keras = 2.6.0
opencv-contrib-python = 3.4.11.45
opencv-python = 3.4.11.45
'''

import cv2
import sys
from keras.utils.data_utils import get_file
from sklearn.model_selection import train_test_split
from keras import layers, models
from keras.applications.vgg16 import VGG16
from keras import Input
from keras import optimizers, initializers, regularizers, metrics
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from time import time
from sklearn.model_selection import KFold, StratifiedKFold
from keras.preprocessing import image
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from ast import literal_eval
import datetime
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization
from keras import backend as K

CAM_ID=0


#얼굴인식
TRACKING_STATE_CHECK = 0
#얼굴인식 위치를 기반으로 추적 기능 초기화
TRACKING_STATE_INIT = 1
#추적 동작
TRACKING_STATE_ON = 2

(major_ver,minor_ver,subminor_ver)=(cv2.__version__).split('.')

#모델 로드
model = tf.keras.models.load_model('./Model-fold_2-37-3.21-1.00.5h')

#예측 분류 리스트
prediction_classes = ["m_10", "m_20", "m_30", "m_40", "m_50", "m_60", "m_70", "f_10", "f_20", "f_30", "f_40", "f_50", "f_60", "f_70"]

if __name__=='__main__':
   #cv2 version 확인
   #print((cv2.__version__).split('.'))


   tracker_types=['BOOSTING','MIL','KCF','TLD','MEDIANFLOW','GOTURN']
   tracker_type = tracker_types[2]

   if int(minor_ver)<3:
       tracker = cv2.Tracker_create(tracker_type)
   else:

       if tracker_type =='BOOSTING':
           tracker=cv2.TrackerBoosting_create()
       if tracker_type =='MIL':
           tracker=cv2.TrackerBoosting_create()
       if tracker_type =='KCF':
           tracker=cv2.TrackerBoosting_create()
       if tracker_type =='TLD':
           tracker=cv2.TrackerBoosting_create()
       if tracker_type =='MEDIANFLOW':
           tracker=cv2.TrackerBoosting_create()
       if tracker_type =='GOTURN':
           tracker=cv2.TrackerBoosting_create()

   #내장 카메라는 0, 외부 카메라는 1부터 부여. CAM_ID는 0
   video = cv2.VideoCapture(cv2.CAP_DSHOW)
   #video = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)


   if not video.isOpened():
       print("Could not open video")
       sys.exit()

   #얼굴인식 함수 생성
   face_cascade = cv2.CascadeClassifier()
   #얼굴인식용 haar 불러오기
   face_cascade.load('./haarcascade_frontalface_default.xml')


   TrackingState=0
   TrackingROI=(0,0,0,0)

   while True:
       ok,frame=video.read()

       if not ok:
           break


       if TrackingState == TRACKING_STATE_CHECK:
           #흑백
           grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           #히스토그램 평활화(재분할)
           grayframe = cv2.equalizeHist(grayframe)
           #얼굴 인식
           faces = face_cascade.detectMultiScale(grayframe, 1.1, 5, 0, (30, 30))


           if len(faces) > 0:
               #얼굴 인식된 위치 및 크기
               for (x,y,w,h) in faces:
                   #TrackingROI에 저장
                   TrackingROI = (x,y,w,h)
                   #순식간에 지나가서 거의 볼수 없음(녹색)
                   cv2.rectangle(frame,(int(x - 0.2 * w), int(y - 0.2 * h)),(int(x + 1.2 * w), int(y + 1.2 * h)),(0,255,0),3, 4, 0)



               #추적 상태를 추적 초기화로 변경
               TrackingState = TRACKING_STATE_INIT
               print('det w : %d ' % w + 'h : %d ' % h)


       #추적 초기화
       #얼굴이 인식되면 동작
       elif TrackingState == TRACKING_STATE_INIT:
           #추적 함수 초기화
           #얼굴인식으로 가져온 위치와 크기를 함께 넣어
           ok = tracker.init(frame, TrackingROI)
           if ok:
               #성공 -> 추적 동작상태로 변경
               TrackingState = TRACKING_STATE_ON
               print('tracking init succeeded')
           else:
               #실패 -> 얼굴 인식상태로 다시
               TrackingState = TRACKING_STATE_CHECK
               print('tracking init failed')


       #추적 동작
       elif TrackingState == TRACKING_STATE_ON:
           #추적
           ok, TrackingROI = tracker.update(frame)
           if ok:
               #추적 성공하면
               #if (TrackingROI[2]>TrackingROI[3]): TrackingROI[3] = TrackingROI[2]
               #else: TrackingROI[2] = TrackingROI[3]
               x, y, w, h = TrackingROI[0], TrackingROI[1], TrackingROI[2], TrackingROI[3]

               #화면에 박스로 표시 (파랑)
               #frame = cv2.equalizeHist(frame)
               cv2.rectangle(frame, (int(x - 0.2 * w), int(y - 0.2 * h)), (int(x + 1.2 * w), int(y + 1.2 * h)), (255,0,0), 2, 1)

               #이미지 크롭
               #face_crop = frame[p1[1]:p2[1], p1[0]:p2[0]]

               # 정사각형으로 크롭되도록 조건문 붙여서 수정
               #if (w == 0): w = 10
               #if (h == 0): h = 10
               if (w > h):
                   face_crop = frame[int(y - 0.2 * w): int(y + 1.2 * w), int(x - 0.2 * w): int(x + 1.2 * w)]
               else:
                   face_crop = frame[int(y - 0.2 * h): int(y + 1.2 * h), int(x - 0.2 * h): int(x + 1.2 * h)]

               #print(face_crop.shape)
               if((face_crop.shape[0]!=0) and (face_crop.shape[1]!=0)):
                   face_crop_resize = cv2.resize(face_crop, (200, 200), interpolation=cv2.INTER_AREA)

                   face_crop_resize = face_crop_resize / 255
                   face_array = image.img_to_array(face_crop_resize)
                   face_dims = np.expand_dims(face_array, axis=0)
                   # print(face_dims.shape)

                   # age, gender 예측
                   prediction = model.predict(face_dims)
                   # print(prediction)

                   # 화면에 박스 위로 결과 표시
                   result = np.argmax(prediction)
                   print(prediction_classes[result])
                   cv2.putText(frame, prediction_classes[result], (int(x - 0.5 * w), int(y - 0.5 * h)),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 225, 0), 2)


               #print('success x %d ' % (int(TrackingROI[0])) + 'y %d ' % (int(TrackingROI[1])) +
               #      'w %d ' % (int(TrackingROI[2])) + 'h %d ' % (int(TrackingROI[3])))

               #print('success x %d ' % (int(TrackingROI[0])) + 'y %d ' % (int(TrackingROI[1])) + 'w %d ' % (int(TrackingROI[2])) + 'h %d ' % (int(TrackingROI[3])))
               #print(prediction)
           else:
               print('Tracking failed')

               TrackingState = TRACKING_STATE_CHECK


       #화면에 카메라 영상 표시
       #추적된 박스 있으면 같이 표시
       cv2.imshow("Tracking",frame)


       #ESC키 = 종료
       k= cv2.waitKey(1) & 0xff
       if k == 27:
           break



'''import tensorflow as tf
hello = tf.constant("Hello, TensorFlow!")
sess = tf.compat.v1.Session()
print(sess.run(hello))'''
