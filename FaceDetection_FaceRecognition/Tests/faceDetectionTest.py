import numpy as np
import cv2
import sys
from PIL import Image
import os
import time
from matplotlib import pyplot as plt
from shutil import copyfile

#stop time for diagnostic reasons
startTime = time.time()

#location of folders with images in each subfolder
_imageSourcePath    = '../data/yaleB/ErrorYaleB/Data'
#path to openCV haarcascade xml-files - CHANGE WITH CAUTION
_cascadePath        = 'C:/opencv/data/haarcascades/'
#path where to copy all faces which failed to detect
#can be overridden (input-prompt after script completed)
_failedFacesPath    = _imageSourcePath + '_FAILED'

#load xml haarcascade for face
face_cascade        = cv2.CascadeClassifier(_cascadePath + 'haarcascade_frontalface_default.xml')



#exit if haarcascade hasnt been loaded
if face_cascade.empty():
    print('Failed to load haarcascade_frontalface_default.xml')
    sys.exit(0)

#get total number of images that will be processed
totalImages = 0
for folder in os.listdir(_imageSourcePath):
    for img in os.listdir(_imageSourcePath + '/' + folder):
        if img.endswith('.info'):
            continue
        else:
            totalImages += 1

def histogram_equalization(img, show=False):
    equ = cv2.equalizeHist(img)
    if show:
        res = np.hstack((img,equ)) #stacking images side-by-side
        cv2.imshow('img',res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return equ


#try to detect a face in each image
#counts processed images
counter = 0
#counts number of detectes faces
faceDetected = 0
#counts number of non detected faces
noFaceDetected = 0
#list of images where no face has been detected
noFaceList = list()
for folder in os.listdir(_imageSourcePath):
    currPath = _imageSourcePath + '/' + folder
    for img in os.listdir(currPath):
        try:
            print(counter, '/', totalImages,  end='\r')
            #print(counter)
            counter += 1
            image = cv2.imread(currPath + '/' + img, 0)
            image = histogram_equalization(image)
            faces = face_cascade.detectMultiScale(image, 1.3, 5)
            if len(faces) == 0:
                #print('no face')
                noFaceDetected += 1
                noFaceList.append(currPath + '/' + img)
                #image = histogram_equalization(image)
                #faces = face_cascade.detectMultiScale(image, 1.3, 5)
                #if len(faces) == 0:
                    #noFaceDetected += 1
                    #noFaceList.append(currPath + '/' + img)

                #else:
                    #faceDetected += 1
            else:
                #print('face')
                faceDetected += 1
            #for (x,y,w,h) in faces:
                #cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                #roi_gray = image[y:y+h, x:x+w]
            #cv2.imshow('img',image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        except Exception as e:
            #print('ERROR:\n', e)
            raise
    break

endTime = time.time()

print('\n')
print('REPORT\nDetected faces\t\t', faceDetected, '\nNot detected faces\t', noFaceDetected)
print('Success:\t\t', (faceDetected/counter)*100, '%')
print('Processed images:\t', counter, '/', totalImages)
print('Elapsed time:\t\t', endTime - startTime)


review = input('Review failed images?\n(y/n)')
if review == 'y':
    for entry in noFaceList:
        print(entry)
    displayImages= input('Display failed images?\n(y/n)')
    if displayImages == 'y':
        for img in noFaceList:
            image = cv2.imread(img, 0)
            equ = cv2.equalizeHist(image)
            res = np.hstack((image,equ)) #stacking images side-by-side
            cv2.imshow(img, res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
copy = input('Copy failed faces?\n(y/n)')
if copy == 'y':
    folderName = input('Folder name (press enter for default):')
    _failedFacesPath    = _imageSourcePath + folderName if not folderName else _failedFacesPath
    if not os.path.exists(_failedFacesPath):
        os.makedirs(_failedFacesPath)
    for img in noFaceList:
        dest = _failedFacesPath + '/' + str(img.split('/')[-1:][0])
        copyfile(img, dest)
