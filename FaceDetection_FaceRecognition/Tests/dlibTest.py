import cv2
import dlib

staticOrVideo = 1 #0=static image, 1 = video stream from webcam

detector    = dlib.get_frontal_face_detector()
predictor   = dlib.shape_predictor('C:\dlib\shape_predictor_68_face_landmarks.dat')


if not staticOrVideo:
    images = ['side1.jpg', 'side2.jpg', 'side3.jpg', 'side4.jpg', 'side5.jpg', 'side6.jpg', 'side7.jpg']
    def showLandmarks(img):
        image       = cv2.imread('../data/test/side/' + img)
        image_gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_gray  = cv2.equalizeHist(image_gray)

        faces = detector(image_gray, 1)

        if not faces:
            print('no face detected in', img)

        for k,d in enumerate(faces):
            shape = predictor(image_gray, d)
            for i in range(1,68):
                cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2)

        cv2.imshow(img, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    for img in images:
        showLandmarks(img)

if staticOrVideo:
    #Code from : http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)

        detections = detector(clahe_image, 1) #Detect the faces in the image

        for k,d in enumerate(detections): #For each detected face

            shape = predictor(clahe_image, d) #Get coordinates
            for i in range(1,68): #There are 68 landmark points on each face
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame

        cv2.imshow("image", frame) #Display the frame

        if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
            break
