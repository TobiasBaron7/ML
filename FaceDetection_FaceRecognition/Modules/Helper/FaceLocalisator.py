import cv2
import sys

# path to openCV haarcascade xml-files - CHANGE WITH CAUTION
_cascadePath        = 'C:/opencv/data/haarcascades/'

# load xml haarcascade for face only if needed
_face_cascade       = ''


def haarcascades_detection(img, num_of_faces=100, scaleFactor = 1.3, minNeighbors=5):
    """
    Uses OpenCV pre-trained haarcascade-classifier to detect faces in the given image.

    :param img:             image, in which to search for faces
    :param num_of_faces:    max number of faces returned, default is 100
    :param scaleFactor:     specifies how much the image size is reduced at each image scale
    :param minNeighbours:   specifies how many neighbors each candidate rectangle should have to retain it
    :return:                list of rectangular faces
    """
    global _face_cascade

    if not _face_cascade:
        _face_cascade = cv2.CascadeClassifier(_cascadePath + 'haarcascade_frontalface_default.xml')

        # exit if haarcascade hasn't been loaded
        if _face_cascade.empty():
            print(
                'ERROR: Modules.Helper.FaceLocalisator: Failed to load haarcascades_frontalface_default.xml from ' + _cascadePath)
            sys.exit(0)

    faces = _face_cascade.detectMultiScale(image=img, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    if faces.shape[0] < num_of_faces:
        return faces

    else:
        # TODO reduce number of faces  (take biggest ones/ nearest to center/...)
        pass
