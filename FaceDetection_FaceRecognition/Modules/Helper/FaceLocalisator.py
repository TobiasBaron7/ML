import cv2
import sys
import dlib

# path to openCV haarcascade and dlib files - CHANGE WITH CAUTION
_cascadePath        = 'C:/opencv/data/haarcascades/'

# load objects only if needed
_face_cascade       = ''
_dlib_detector      = ''


def _select_n_faces(face_list, n):
    """
    Select n biggest faces of given face_list and remove the rest.
    Size is calculated by width + length.

    :param face_list:   list of faces, from where to remove
    :param n:           faces are removed until n faces are left
    """
    face_list.sort(key=lambda x: x.shape[0] + x.shape[1], reverse=True)
    while len(face_list) > n:
        face_list.pop(len(face_list)-1)


def haarcascades_detection(img, max_faces=100, scale_factor=1.3, min_neighbors=5):
    """
    Uses OpenCV pre-trained haarcascade-classifier to detect faces in the given image.

    :param img:             image, in which to search for faces
    :param max_faces:    max number of faces returned, default is 100
    :param scale_factor:     specifies how much the image size is reduced at each image scale
    :param min_neighbors:   specifies how many neighbors each candidate rectangle should have to retain it
    :return:                list of rectangular faces
    """
    global _face_cascade

    if not max_faces:
        max_faces = 100
    if not scale_factor:
        scale_factor = 1.3
    if not min_neighbors:
        min_neighbors = 5

    if not _face_cascade:
        _face_cascade = cv2.CascadeClassifier(_cascadePath + 'haarcascade_frontalface_default.xml')

        # exit if haarcascade hasn't been loaded
        if _face_cascade.empty():
            print('ERROR: Modules.Helper.FaceLocalisator: Failed to load haarcascades_frontalface_default.xml from '
                  + _cascadePath)
            sys.exit(0)

    faces = _face_cascade.detectMultiScale(image=img,
                                           scaleFactor=scale_factor,
                                           minNeighbors=min_neighbors)

    # crop faces out of image and append to list
    face_list = list()
    for x, y, w, h in faces:
        face_list.append(img[y:y+w, x:x+h])

    if faces.shape[0] > max_faces:
        _select_n_faces(face_list, max_faces)

    return face_list


def hog_detection(img, max_faces=100, up_sampling=1):
    """
    Detects faces on given grayscale image using histogram orientated gradients.

    up_ sampling defines the times the image is increased in size in order to detect even
    small faces. Computation time increases as up_sampling increases.

    :param img:         grayscale image
    :param max_faces:   maximum number of faces to detect
    :param up_sampling: number of times image is up-sampled
    :return:            list of rectangular faces or empty list if none found
    """
    global _dlib_detector

    if not max_faces:
        max_faces = 100
    if not up_sampling:
        up_sampling = 1

    if not _dlib_detector:
        _dlib_detector  = dlib.get_frontal_face_detector()

        if not _dlib_detector:
            if not _dlib_detector:
                print('ERROR: Modules.FaceLocalisator: Failed to instantiate dlib detector.')
                sys.exit(0)

    faces = _dlib_detector(img, up_sampling)

    face_list = list()
    for face in faces:
        face_list.append(img[face.top():face.bottom(), face.left():face.right()])

    if len(faces) > max_faces:
        _select_n_faces(face_list, max_faces)

    return face_list
