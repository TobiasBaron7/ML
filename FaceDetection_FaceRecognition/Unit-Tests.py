import cv2
import Modules.Helper.ImageProcessor as ip
import Modules.Helper.FaceLocalisator as fl

# General settings
image_path              = 'data/test/side/side7.jpg'

# Test-Settings
# define which module to test
test_ImageProcessor     = False
gamma                   = 3
cliplimit               = 20.0
tileGridSize            = (8, 8)

test_FaceLocalisator    = True
scaleFactor             = 1.25
minNeighbors            = 4


# TEST ImageProcessor
# methods: img2gray(), histogram_equalization(),
# adaptive_histogram_equalization(), gamma_correction()
if test_ImageProcessor:
    img_original    = cv2.imread(image_path)

    img_original    = ip.img2gray(img_original)
    img_gamma       = ip.gamma_correction(img_original, gamma=gamma)
    img_hist        = ip.histogram_equalization(img_original)
    img_adapt_hist  = ip.clahe(img_original, cliplimit=cliplimit, tileGridSize=tileGridSize)

    cv2.imshow('grayscale', img_original)
    cv2.imshow('gamma_correction: ' + str(gamma), img_gamma)
    cv2.imshow('adaptive_histogram_equalization: ' + str(cliplimit) + ', ' + str(tileGridSize), img_adapt_hist)
    cv2.imshow('histogram_equalization', img_hist)

    # call again with different parameters, which force a new object creation for saved parameter-set
    ip.gamma_correction(img_original)
    ip.gamma_correction(img_original, gamma=4)
    ip.clahe(img_original)
    ip.clahe(img_original, cliplimit=10, tileGridSize=(1,50))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TEST FaceLocalisator
if test_FaceLocalisator:
    img_rgb = cv2.imread(image_path)
    img_gray = ip.img2gray(img_rgb)

    faces = fl.haarcascades_detection(img_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    for (x, y, w, h) in faces:
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('img', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

