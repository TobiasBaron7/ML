import numpy as np
import cv2


# prints changes in settings at function calls
_log_changes = True

# if clahe is used, save clahe-object globally to avoid re-instantiating at every call
_clahe          = ''
_cliplimit      = 2.0
_tile_grid_size   = (8, 8)

# if gamma_correction is used, save lookup-table to avoid re-computing at every call
_gamma_lut  = np.empty([0])
_gamma      = 1.0


def img2gray(img):
    """
    Converts given image to grayscale

    :param img:     RGB image
    :return:        Converted grayscale image
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def histogram_equalization(img):
    """
    Equalizes the histogram of a given grayscale image

    :param img:     grayscale img
    :return:        grayscale image with equalized histogram
    """
    return cv2.equalizeHist(img)


def clahe(img, cliplimit=2.0, tile_grid_size=(8, 8)):
    """
    CLAHE: Contrast Linmited Adaptive Histogram Equalization
    The image is divided into small blocks called "tiles".
    Then each of these blocks are histogram equalized as usual.
    If noise is there, it will be amplified. To avoid this, contrast limiting is applied.
    If any histogram bin is above the specified contrast limit,
    those pixels are clipped and distributed uniformly to other bins before applying histogram equalization.
    After equalization, to remove artifacts in tile borders, bilinear interpolation is applied.

    :param img:             grayscale image
    :param cliplimit:       contrast limit of each bin
    :param tile_grid_size:    height x width of area
    :return:                grayscale image with clahe applied
    """
    global _log_changes, _clahe, _cliplimit, _tile_grid_size

    if not _clahe:
        _clahe          = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=tile_grid_size)
        _cliplimit      = cliplimit
        _tile_grid_size   = tile_grid_size

    # create new clahe object if function is called with different parameters than before
    if _cliplimit is not cliplimit or _tile_grid_size is not tile_grid_size:
        if _log_changes:
            print('ImageProcessor: clahen - parameters changed\n'
                  + 'cliplimit: %s -> %s\n' % (str(_cliplimit), str(cliplimit))
                  + 'tileGridSize %s -> %s' % (str(_tile_grid_size), str(tile_grid_size)))
        _clahe          = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=tile_grid_size)
        _cliplimit      = cliplimit
        _tile_grid_size   = tile_grid_size

    return _clahe.apply(img)


# from https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def gamma_correction(img, gamma=1.0):
    """
    Applies a gamma correction to given grayscale image.

    :param img:     grayscale image
    :param gamma:   gamma value
    :return:        Image with gamma correction applied
    """
    global _log_changes, _gamma_lut, _gamma

    inv_gamma = 1.0 / gamma
    # create a lookup-table with 256 entries (index=0,...,255)
    # with table[x] = ((x/255)^inv_gamma) * 255
    # then applies table to image, where each pixel of the image with pixel-value = x
    # gets a new value which is table[x]
    if not _gamma_lut.size:
        _gamma_lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
        _gamma = gamma

    # create new lookup-table if function is called with different gamma value than before
    if _gamma is not gamma:
        if _log_changes:
            print('ImageProcessor: gamma_correction - parameter changed\n'
                  + 'gamma: %s -> %s' % (str(_gamma), str(gamma)))
        _gamma_lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
        _gamma = gamma

    return cv2.LUT(img, _gamma_lut)


"""
SETTER
"""


def set_log_changes(b):
    global _log_changes
    _log_changes = b
