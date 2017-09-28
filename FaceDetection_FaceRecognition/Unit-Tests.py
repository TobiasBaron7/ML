import cv2
import Modules.Helper.ImageProcessor as ip
import Modules.Helper.FaceLocalisator as fl

# General settings
image_path                = 'data/test/family.jpg'

# Test-Settings
# define which module to test
test_ImageProcessor     = False
gamma                   = 3
cliplimit               = 20.0
tile_grid_size          = (8, 8)

test_FaceLocalisator     = True
method_to_test           = 0    # 0=haarcascades, 1=HOG
scale_factor             = 1.25
min_neighbors            = 4
max_faces                = 3
up_sampling              = 0


# TEST ImageProcessor
# methods: img2gray(), histogram_equalization(),
# adaptive_histogram_equalization(), gamma_correction()
if test_ImageProcessor:
    img_original    = cv2.imread(image_path)

    img_original    = ip.img2gray(img_original)
    img_gamma       = ip.gamma_correction(img_original, gamma=gamma)
    img_hist        = ip.histogram_equalization(img_original)
    img_adapt_hist  = ip.clahe(img_original, cliplimit=cliplimit, tile_grid_size=tile_grid_size)

    cv2.imshow('grayscale', img_original)
    cv2.imshow('gamma_correction: ' + str(gamma), img_gamma)
    cv2.imshow('adaptive_histogram_equalization: ' + str(cliplimit) + ', ' + str(tile_grid_size), img_adapt_hist)
    cv2.imshow('histogram_equalization', img_hist)

    # call again with different parameters, which force a new object creation for saved parameter-set
    ip.gamma_correction(img_original)
    ip.gamma_correction(img_original, gamma=4)
    ip.clahe(img_original)
    ip.clahe(img_original, cliplimit=10, tile_grid_size=(1, 50))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TEST FaceLocalisator
if test_FaceLocalisator:
    img_rgb = cv2.imread(image_path)
    img_gray = ip.img2gray(img_rgb)

    # OpenCV haarcascades
    if method_to_test is 0:
        faces = fl.haarcascades_detection(img_gray,
                                          scale_factor=scale_factor,
                                          min_neighbors=min_neighbors,
                                          max_faces=max_faces)

        counter = 0
        for face in faces:
            cv2.namedWindow('face ' + str(counter), cv2.WINDOW_NORMAL)
            cv2.imshow('face ' + str(counter), face)
            counter += 1

    # dlib histogram of orientated gradients
    if method_to_test is 1:
        faces = fl.hog_detection(img_gray,
                                 max_faces=max_faces,
                                 up_sampling=up_sampling)
        counter = 0
        for face in faces:
            cv2.namedWindow('face ' + str(counter), cv2.WINDOW_AUTOSIZE)
            cv2.imshow('face ' + str(counter), face)
            counter += 1


    cv2.waitKey(0)
    cv2.destroyAllWindows()
