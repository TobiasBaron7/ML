import numpy as np
import cv2
import sys

cascadePath = 'C:/opencv/data/haarcascades/'

image = '../data/test/happy.jfif'

xmlLoadError = False


face_cascade    = cv2.CascadeClassifier(cascadePath + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print('Failed to load haarcascade_frontalface_default.xml')
    xmlLoadError = True

eye_cascade     = cv2.CascadeClassifier(cascadePath + 'haarcascade_eye.xml')
if face_cascade.empty():
    print('Failed to load haarcascade_eye.xml')
    xmlLoadError = True

if xmlLoadError:
    sys.exit(0)

img             = cv2.imread(image)
gray            = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(gray, 1.3, 5)


for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
