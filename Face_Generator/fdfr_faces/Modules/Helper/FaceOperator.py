import dlib
import sys
import cv2
from skimage import filters


_dlib_path          = 'C:\dlib\shape_predictor_68_face_landmarks.dat'
_dlib_predictor     = ''


def get_landmarks(img):
    """
    Computates landmarks for the face in given image.
    area_of_interest is the complete image as method expects to only receive
    already cropped images, where only the face is left.

    :param img: grayscale image of cropped face
    :return:    set of landmark points
    """
    global _dlib_path, _dlib_predictor

    if not _dlib_predictor:
        _dlib_predictor = dlib.shape_predictor(_dlib_path)
        if not _dlib_predictor:
            print('ERROR: Modules.Helper.FaceOperator: Failed loading dlib shape_predictor from', _dlib_path)
            sys.exit(0)

    area_of_interest = dlib.rectangle(0, 0, img.shape[0], img.shape[1])

    return _dlib_predictor(img, area_of_interest)


def scale(img, width, height):
    """
    Scales the given image to fit given width and height.

    :param img:     original image
    :param width:   new width in pixels
    :param height:  new height in pixels
    :return:        scaled image
    """
    if not width and not height:
        return
    if not width:
        width = height
    if not height:
        height = width

    return cv2.resize(img, (width, height))


# TODO
def frontalization():
    pass


# TODO
def equalize_background(img):
    return cv2.Canny(img, 30, 200)



